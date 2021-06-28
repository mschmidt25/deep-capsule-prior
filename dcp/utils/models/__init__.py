from dcp.utils.models.skip import Skip
from dcp.utils.models.skip_capsule import SkipCapsule


def get_skip_model(in_ch=1,
                   out_ch=1,
                   channels=(128,) * 5,
                   skip_channels=(4,) * 5):
    return Skip(in_ch=in_ch,
                out_ch=out_ch,
                channels=channels,
                skip_channels=skip_channels)

def get_capsule_skip_model(in_ch=1,
                           out_ch=1, 
                           capsules=[[8, 2], [16, 2], [32, 2], [64, 2]],
                           skip_capsules=[[4, 2], [4, 2], [4, 2], [4, 2]],
                           iter_rout=1,
                           weight_init='xavier_uniform',
                           same_filter=False,
                           use_bias=True,
                           eps=1e-6,
                           op=None):
    return SkipCapsule(in_ch=in_ch,
                       out_ch=out_ch,
                       capsules=capsules,
                       skip_capsules=skip_capsules,
                       iter_rout=iter_rout,
                       weight_init=weight_init,
                       same_filter=same_filter,
                       use_bias=use_bias,
                       eps=eps,
                       op=op)
