import torch
import torch.nn as nn

from dcp.utils.layers.caps_layers import PrimaryConvCaps2d, ConvCaps2d, ConvCapsBN
from dcp.utils.layers.caps_routing import DirectRouting
from dcp.utils.layers.fbp_layers import FBPModule


class SkipCapsule(nn.Module):
    def __init__(self, in_ch, out_ch, 
                 capsules=[[8, 2], [16, 2], [32, 2], [64, 2]],
                 skip_capsules=[[4, 2], [4, 2], [4, 2], [4, 2]],
                 iter_rout=1, weight_init='xavier_uniform', same_filter=False,
                 use_bias=True, eps=1e-6, op=None):
        super(SkipCapsule, self).__init__()
        
        self.scales = len(capsules)
        assert(len(capsules) == len(skip_capsules))
        
        self.primary_capsule = PrimaryConvCaps2d(in_channels=in_ch,
                                 out_caps=capsules[0], kernel_size=3, stride=1,
                                 padding=1, weight_init=weight_init)
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        
        if op is not None:
            self.fbp_op = FBPModule(op, filter_type='Hann',
                     frequency_scaling=0.7)
        else:
            self.fbp_op = None

        cur_caps = capsules[0]
        
        capsules = [cur_caps] + capsules
        
        for i in range(self.scales):
            self.down.append(nn.Sequential(
                CapsuleBlock(in_caps=cur_caps, out_caps=capsules[i+1],
                             kernel_size=3, stride=2, iter_rout=iter_rout,
                             weight_init=weight_init, same_filter=same_filter,
                             use_bias=use_bias, eps=eps),
                CapsuleBlock(in_caps=capsules[i+1],out_caps=capsules[i+1],
                             kernel_size=1, stride=1, iter_rout=iter_rout,
                             eps=eps)))
            cur_caps = capsules[i+1]

        for i in range(self.scales):
            self.up.append(UpCapsuleBlock(in_caps=cur_caps,
                                          out_caps=capsules[-i-1],
                                          skip_in_c=capsules[-i-2],
                                          skip_out_c=skip_capsules[-i-1],
                                          kernel_size=3, iter_rout=iter_rout,
                                          weight_init=weight_init,
                                          same_filter=same_filter,
                                          use_bias=use_bias,
                                          eps=eps))
            cur_caps = capsules[-i-1]

        self.outc = CapsuleBlock(in_caps=capsules[0], out_caps=[out_ch, 1],
                                 kernel_size=1, iter_rout=iter_rout,
                                 weight_init=weight_init,
                                 same_filter=same_filter,
                                 use_bias=use_bias,
                                 eps=eps)

    def forward(self, x0):
        
        if self.fbp_op is not None:
            x0 = self.fbp_op(x0)

        a, v = self.primary_capsule(x0)
        #print("Primary Capsule, x0, a, v: ", x0.shape, a.shape, v.shape)
        xs = [[a, v]]
        for i in range(self.scales):
            a_temp, v_temp = self.down[i](xs[-1])
            #print("Capsule down {}. Size: a {} , v {}".format(i, a_temp.shape, v_temp.shape))
            xs.append([a_temp, v_temp])
        a, v = xs[-1]

        for i in range(self.scales):
            #print("Input to Up-Capsule at scale {}. a: {}, v: {}".format(i, a.shape, v.shape))
            #print("Input to Up-Capsule at scale {}. Skip: a: {}, v:{}".format(i, xs[-2-i][0].shape, xs[-2-i][1].shape))
            a, v = self.up[i]([[a, v], xs[-2-i]])
            #print("Output of Up-Capsule at scale {}. a: {}, v:{}".format(i, a.shape, v.shape))
        # a: [?, C, F, F], v: [?, C, 1, 1, F, F]
        a, v = self.outc([a, v])
        #print("Out-Capsule. a: {}, v: {}".format(a.shape, v.shape))
        v_s = v.shape
        # (?, C, F, F)
        v = v.reshape(-1, v_s[1], v_s[-2], v_s[-1])
        
        v = torch.sigmoid(v)

        return  a,v


class CapsuleBlock(nn.Module):
    def __init__(self, in_caps, out_caps, kernel_size=3, stride=1, 
                 iter_rout=1, same_filter=False, weight_init='xavier_uniform',
                 use_bias=True, eps=1e-6):
        super(CapsuleBlock, self).__init__()
                        
        to_pad = int((kernel_size - 1) / 2)

        self.conv_capsule = ConvCaps2d(in_caps=in_caps, out_caps=out_caps,
                               kernel_size=kernel_size, stride=stride, 
                               padding=to_pad, weight_init=weight_init,
                               same_filter=same_filter, use_bias=use_bias)
        
        self.bn = ConvCapsBN(in_caps=in_caps, out_caps=out_caps)
        self.bn1 = torch.nn.BatchNorm3d(num_features=out_caps[0])
        
        self.act = DirectRouting(in_caps=in_caps, out_caps=out_caps,
                                 iter_rout=iter_rout, eps=1e-6)

    def forward(self, inputs):
        a = inputs[0]
        v = inputs[1]
        a, v = self.conv_capsule([a, v])
        a, v = self.bn([a, v])
        a, v = self.act([a, v])
        v = self.bn1(v)
        return a, v

    
class UpCapsuleBlock(nn.Module):
    def __init__(self, in_caps, out_caps, skip_in_c, skip_out_c, kernel_size=3,
                 iter_rout=1, same_filter=False, weight_init='xavier_uniform',
                 use_bias=True, eps=1e-6):
        super(UpCapsuleBlock, self).__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=False)
        
        self.conv_caps = nn.Sequential(
                            CapsuleBlock(in_caps=[in_caps[0] + skip_out_c[0],
                                                  in_caps[1]],
                                         out_caps=out_caps,
                                         kernel_size=kernel_size,
                                         stride=1, iter_rout=iter_rout,
                                         weight_init=weight_init,
                                         same_filter=same_filter,
                                         use_bias=use_bias,
                                         eps=eps),
                            CapsuleBlock(in_caps=out_caps, out_caps=out_caps,
                                         kernel_size=1, stride=1,
                                         iter_rout=iter_rout,
                                         weight_init=weight_init,
                                         same_filter=same_filter,
                                         use_bias=use_bias,
                                         eps=eps))
        
        self.skip = skip_out_c[0] > 0
        

        if self.skip:
            self.skip_conv_caps = CapsuleBlock(in_caps=skip_in_c,
                                               out_caps=skip_out_c,
                                               kernel_size=1, stride=1,
                                               iter_rout=iter_rout,
                                               weight_init=weight_init,
                                               same_filter=same_filter,
                                               use_bias=use_bias,
                                               eps=eps)
    
            self.concat_caps = CapsuleConcat()
            
        

    def forward(self, inputs):
        # Upsampling route
        x0 = inputs[0]
        
        a0, v0 = x0      
        v0s = v0.shape
        a0 = self.up(a0)
        
        # Reshape for upsampling
        # [?, C*P, F0_old, F1_old] 
        v0 = self.up(v0.reshape(-1, v0s[1]*v0s[2], v0s[-2], v0s[-1]))
        
        # [?, C, P, F0_new, F1_new]
        v0 = v0.reshape(-1, v0s[1], v0s[2], v0.shape[-2], v0.shape[-1])
        
        # Skip route
        x1 = inputs[1]
        a1, v1 = x1
        v1_s = v1.shape
        
        if self.skip:
            # Add the skip connection
            #print("Skip: ")
            a1, v1 = self.skip_conv_caps([a1, v1])
            #print("Out of skip_conv_caps. a: {}, v: {}".format(a1.shape, v1.shape))
            a, v = self.concat_caps([[a0, v0], [a1, v1]])
            #print("Input to concat caps. a0: {}, v0: {}, a1: {}: , v1: {}".format(a0.shape, v0.shape, a1.shape, v1.shape))
            #print("----")
        else:
            # Adjust dimension of the upsampled version to guarantee the same
            # spatial dimension as during downsampling
            # a: [?, B, F0_a1, F1_a1], v: [?, B, P, F0_v1, F1_v1]
            a, v = adapt_size([a0, v0], [v1_s[-2], v1_s[-1]])
        #print("Conv Caps in Up-Sampling: a: {}, v: {}".format(a.shape, v.shape))
        a, v = self.conv_caps([a, v])
        #print("Output of conv caps: a: {}, v: {}".format(a.shape, v.shape))
        return a, v


def adapt_size(inputs, new_shape):
    """
    Center crop the spatial size of capsules to the desired area

    Parameters
    ----------
    inputs : list of tensor
        Activation and pose tensor.
    new_shape : list of int
        New spatial height and width.

    Returns
    -------
    a : Tensor
        Center cropped activity.
    v : TYPE
        Center cropped pose.

    """
    a = inputs[0]
    v = inputs[1]
    
    v_s = v.shape
    
    target_F0 = new_shape[0]
    target_F1 = new_shape[1]

    diff = [(v_s[-2] - target_F0) // 2, (v_s[-1] - target_F1) // 2]
            
    a = a[:, :, diff[0]: diff[0] + target_F0, diff[1]: diff[1] + target_F1]
    v = v[:, :, :, diff[0]: diff[0] + target_F0, diff[1]: diff[1] + target_F1]
    
    return a, v


class CapsuleConcat(nn.Module):
    def __init__(self):
        super(CapsuleConcat, self).__init__()

    def forward(self, inputs):
        caps0 = inputs[0]
        caps1 = inputs[1]
        
        # m: [?, C, P, F0, F1] 
        c0_m_shape = caps0[1].shape
        c1_m_shape = caps1[1].shape
        
        a0 = caps0[0]
        a1 = caps1[0]
        v0 = caps0[1]
        v1 = caps1[1]
        
        # Check for spatial dimension match. Reduce to smaller size if unequal
        if c0_m_shape[-2] == c1_m_shape[-2] and \
            c0_m_shape[-1] == c1_m_shape[-1]:
            
            target_F0 = c1_m_shape[-2]
            target_F1 = c1_m_shape[-1]

        else:
            target_F0 = min(c0_m_shape[-2], c1_m_shape[-2])
            target_F1 = min(c0_m_shape[-1], c1_m_shape[-1])

            a0, v0 = adapt_size([a0, v0], [target_F0, target_F1])
            a1, v1 = adapt_size([a1, v1], [target_F0, target_F1])
            
        # a: [?, C0 + C1, F0, F1] 
        a = torch.cat([a0, a1], dim=1)
        
        # v: [?, C*P, F0, F1] 
        v0 = v0.reshape(-1, c0_m_shape[1]*c0_m_shape[2],
                        target_F0, target_F1)
        v1 = v1.reshape(-1, c1_m_shape[1]*c1_m_shape[2],
                        target_F0, target_F1)
        
        # v: [?, (C0+C1)*P, F0, F1]
        v = torch.cat([v0, v1], dim=1)
        
        # v: [?, C0+C1, P, F0, F1]
        v = v.reshape(-1, c0_m_shape[1]+c1_m_shape[1], c0_m_shape[2],
                      target_F0, target_F1)        
            
        return a, v
    
