import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectRouting(nn.Module):
    '''Direct Routing for Capsule Layers'''
    def __init__(self, in_caps, out_caps, iter_rout=3, eps=1e-6):
        
        super().__init__()
        
        self.A = in_caps[0]
        self.B = out_caps[0]
        self.P = out_caps[1]
        
        self.iter_rout = iter_rout
        
        self.eps = eps
        
        logits = torch.ones(1, self.A, self.B, 1, 1, 1)
        self.logits = nn.Parameter(logits, requires_grad=True)

    def forward(self, inputs):
        # a: [?, A, F_in0, F_in1], v: [?, A, B, P_out, F_out0, F_out1]
        a = inputs[0]
        v = inputs[1]
        
        v_s = v.shape
        
        # Initial value for the logits. Create copies for all batches.
        # b: [?, A, B, 1, 1, 1]
        b = torch.repeat_interleave(self.logits, repeats=v_s[0], dim=0)
        
        for i in range(self.iter_rout):
            # 1) Calculate softmax of current logits
            # c: [?, A, B, 1, 1, 1]
            #c = b 
            c = F.softmax(b, dim=2)
            
            # 2) Weighting of the input poses and summing over the A dimension
            # s: [?, 1, B, P_out, F_out0, F_out1]
            s = torch.sum(c*v, dim=1, keepdims=True)
            
            # 3) Apply squash function
            # p: [?, 1, B, P_out, F_out0, F_out1]
            p = self.squash(s)
            
            # 4) Update logits
            if i < (self.iter_rout - 1):
                # b: [?, A, B, 1, 1, 1]
                b = b + torch.sum(p*v, dim=(-3,-2,-1), keepdim=True)
        
        # p: [?, B, P_out, F_out0, F_out1]
        p = p.squeeze(dim=1)
        
        # a: [?, B, F_out0, F_out1]
        a = torch.linalg.norm(p, ord=2, dim=2)    
        
        return a, p

    def squash(self, s):
        # norm_s_sq: [?, 1, B, 1, F_out0, F_out1]
        norm_s_sq = torch.sum(torch.pow(s, exponent=2), dim=2, keepdim=True)
        
        p = torch.sqrt(norm_s_sq + self.eps) / (1.0 + norm_s_sq) * s
        
        return p
