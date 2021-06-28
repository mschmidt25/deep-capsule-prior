import torch
import torch.nn as nn
import torch.nn.functional as F

class PrimaryConvCaps2d(nn.Module):
    '''Primary Capsule Layer'''
    def __init__(self, in_channels, out_caps, kernel_size, stride,
        padding=0, weight_init='xavier_uniform'):
        
        super().__init__()
        
        # Check for tuple
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride)

        self.A = in_channels
        self.B = out_caps[0]
        self.P = out_caps[1]
        self.K = kernel_size
        self.S = stride
        self.padding = padding

        p_kernel = torch.empty(self.B*self.P, self.A, self.K[0], self.K[1])
        a_kernel = torch.empty(self.B, self.A, self.K[0], self.K[1])

        if weight_init == 'kaiming_normal':
            nn.init.kaiming_normal_(p_kernel)
            nn.init.kaiming_normal_(a_kernel)
        elif weight_init == 'kaiming_uniform':
            nn.init.kaiming_uniform_(p_kernel)
            nn.init.kaiming_uniform_(a_kernel)
        elif weight_init == 'xavier_normal':
            nn.init.xavier_normal_(p_kernel)
            nn.init.xavier_normal_(a_kernel)
        elif weight_init == 'xavier_uniform':
            nn.init.xavier_uniform_(p_kernel)
            nn.init.xavier_uniform_(a_kernel)
        else:
            NotImplementedError('{} not implemented.'.format(weight_init))

        # Out ← [B*(P+1), A, K0, K1]
        self.weight = nn.Parameter(torch.cat([a_kernel, p_kernel], dim=0))

        self.BN_a = nn.BatchNorm2d(self.B, affine=True)
        self.BN_p = nn.BatchNorm3d(self.B, affine=True)

    def forward(self, x): # [?, A, F_in0, F_in1] ← In

        # [?, B*(P+1), F_out0, F_out1]
        x = F.conv2d(x, weight=self.weight, stride=self.S, 
                     padding=self.padding)

        # ← [?, B*(P+1), F_out0, F_out1]
        # ([?, B*P, F_out0, F_out1], [?, B, F_out0, F_out1]) 
        a, p = torch.split(x, [self.B, self.B*self.P], dim=1)

        # [?, B, P, F_out0, F_out1]
        p = self.BN_p(p.reshape(-1, self.B, self.P, *x.shape[2:]))

        # Out ← [?, B, F_out0, F_out1])
        a = torch.sigmoid(self.BN_a(a))

        return (a, p)

class ConvCaps2d(nn.Module):
    '''Convolutional Capsule Layer'''
    def __init__(self, in_caps, out_caps, kernel_size, stride,
        padding=0, weight_init='xavier_uniform', same_filter=False,
        use_bias=True):
        
        super().__init__()
        
        # Check for tuple
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride)

        self.A = in_caps[0]
        self.P_in = in_caps[1]
        self.B = out_caps[0]
        self.P_out = out_caps[1]
        self.K = kernel_size
        self.S = stride
        self.padding = padding
        self.same_filter = same_filter
        self.use_bias = use_bias

        if same_filter:
            out_ch = self.B*self.P_out

        else:
            out_ch = self.B*self.P_out*self.A

        p_kernel = torch.empty(out_ch, self.P_in, self.K[0], self.K[1])
        
        if weight_init == 'kaiming_normal':
            nn.init.kaiming_normal_(p_kernel)
        elif weight_init == 'kaiming_uniform':
            nn.init.kaiming_uniform_(p_kernel)
        elif weight_init == 'xavier_normal':
            nn.init.xavier_normal_(p_kernel)
        elif weight_init == 'xavier_uniform':
            nn.init.xavier_uniform_(p_kernel)
        else:
            NotImplementedError('{} not implemented.'.format(weight_init))

        self.weight = nn.Parameter(p_kernel)
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))
        else:
            self.bias = None


    def forward(self, inputs):
        # a: [?, A, F_in0, F_in1], p: [?, A, P_in, F_in0, F_in1]
        a = inputs[0]
        p = inputs[1]
        
        p_s = p.shape
        
        if self.same_filter:
            # [?*A, P_in, F_in0, F_in1]
            p = p.reshape(p_s[0]*self.A, self.P_in, p_s[3], p_s[4])
            # [?*A, B*P_out, F_out0, F_out1]
            p = F.conv2d(p, weight=self.weight, stride=self.S, bias=self.bias,
                     padding=self.padding)
        else:
            # [?, A*P_in, F_in0, F_in1]
            p = p.reshape(p_s[0], self.A*self.P_in, p_s[3], p_s[4])
            # [?, A*B*P_out, F_out0, F_out1]
            p = F.conv2d(p, weight=self.weight, stride=self.S, bias=self.bias,
                     padding=self.padding, groups=self.A)
        
        # [?, A, B, P_out, F_out0, F_out1]
        p = p.reshape(p_s[0], self.A, self.B, self.P_out,
                      p.shape[-2], p.shape[-1])    
            
        return (a, p)
    
    
class ConvCapsBN(nn.Module):
    '''Batchnorm for Convolutional Capsule Layer'''
    def __init__(self, in_caps, out_caps):
        
        super().__init__()
        
        self.A = in_caps[0]
        self.B = out_caps[0]
        self.P_out = out_caps[1]

        self.bn = torch.nn.BatchNorm3d(num_features=self.B, eps=1e-05,
                                       momentum=0.1, affine=True,
                                       track_running_stats=True)


    def forward(self, inputs):
        # a: [?, A, F_in0, F_in1], p: [?, A, B, P_out, F_out0, F_out1]
        a = inputs[0]
        p = inputs[1]
        
        p_s = p.shape
        
        # p: [?*A, B, P_out, F_out0, F_out1]
        p = self.bn(p.reshape(p_s[0]*self.A, self.B, self.P_out, p_s[-2], p_s[-1]))
        
        # p: [?, A, B, P_out, F_out0, F_out1]
        p = p.reshape(-1, self.A, self.B, self.P_out, p_s[-2], p_s[-1])
            
        return (a, p)
