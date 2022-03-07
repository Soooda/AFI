import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from .FeatureExtractor import FeatureExtractor
from .RFR.rfr_new import RFR
from .ForwardWarp import ForwardWarp
from .GridNet import GridNet

class AnimeInterp(nn.Module):
    def __init__(self, path='models/raft_model/models/rfr_sintel_latest.pth-no-zip', args=None):
        super(AnimeInterp, self).__init__()

        args = argparse.Namespace()
        args.small = False
        args.mixed_precision = False
        args.num_heads = 1
        # args.requires_sq_flow = False

        self.flownet = RFR(args)
        self.feat_ext = FeatureExtractor()
        self.fwarp = ForwardWarp()
        self.synnet = GridNet(6, 64, 128, 96*2, 3)


        if path is not None:
            dict1 = torch.load(path)
            dict2 = dict()
            for key in dict1:
                dict2[key[7:]] = dict1[key]
            self.flownet.load_state_dict(dict2, strict=False)

    def dflow(self, flo, target):
        tmp = F.interpolate(flo, target.size()[2:4])
        tmp[:, :1] = tmp[:, :1].clone() * tmp.size()[3] / flo.size()[3]
        tmp[:, 1:] = tmp[:, 1:].clone() * tmp.size()[2] / flo.size()[2]

        return tmp
    def forward(self, I1, I2, F12i, F21i, t):
        r = 0.6

        # I1 = I1[:, [2, 1, 0]]
        # I2 = I2[:, [2, 1, 0]]


        # extract features
        I1o = (I1 - 0.5) / 0.5
        I2o = (I2 - 0.5) / 0.5

        feat11, feat12, feat13 = self.feat_ext(I1o)
        feat21, feat22, feat23 = self.feat_ext(I2o)

        # calculate motion 

        # with torch.no_grad():
        #     self.flownet.eval()
        F12, F12in, err12, = self.flownet(I1o, I2o, iters=12, test_mode=False, flow_init=F12i)
        F21, F21in, err12, = self.flownet(I2o, I1o, iters=12, test_mode=False, flow_init=F21i)

        F1t = t * F12
        F2t = (1-t) * F21

        F1td = self.dflow(F1t, feat11)
        F2td = self.dflow(F2t, feat21)

        F1tdd = self.dflow(F1t, feat12)
        F2tdd = self.dflow(F2t, feat22)

        F1tddd = self.dflow(F1t, feat13)
        F2tddd = self.dflow(F2t, feat23)

        # warping 

        I1t, norm1 = self.fwarp(I1, F1t)
        feat1t1, norm1t1 = self.fwarp(feat11, F1td)
        feat1t2, norm1t2 = self.fwarp(feat12, F1tdd)
        feat1t3, norm1t3 = self.fwarp(feat13, F1tddd)

        I2t, norm2 = self.fwarp(I2, F2t)
        feat2t1, norm2t1 = self.fwarp(feat21, F2td)
        feat2t2, norm2t2 = self.fwarp(feat22, F2tdd)
        feat2t3, norm2t3 = self.fwarp(feat23, F2tddd)
        
        # normalize
        # Note: normalize in this way benefit training than the original "linear"
        I1t[norm1 > 0] = I1t.clone()[norm1 > 0] / norm1[norm1 > 0]
        I2t[norm2 > 0] = I2t.clone()[norm2 > 0] / norm2[norm2 > 0]
        
        feat1t1[norm1t1 > 0] = feat1t1.clone()[norm1t1 > 0] / norm1t1[norm1t1 > 0]
        feat2t1[norm2t1 > 0] = feat2t1.clone()[norm2t1 > 0] / norm2t1[norm2t1 > 0]
        
        feat1t2[norm1t2 > 0] = feat1t2.clone()[norm1t2 > 0] / norm1t2[norm1t2 > 0]
        feat2t2[norm2t2 > 0] = feat2t2.clone()[norm2t2 > 0] / norm2t2[norm2t2 > 0]
        
        feat1t3[norm1t3 > 0] = feat1t3.clone()[norm1t3 > 0] / norm1t3[norm1t3 > 0]
        feat2t3[norm2t3 > 0] = feat2t3.clone()[norm2t3 > 0] / norm2t3[norm2t3 > 0]


        # synthesis
        It_warp = self.synnet(torch.cat([I1t, I2t], dim=1), torch.cat([feat1t1, feat2t1], dim=1), torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))



        return It_warp, F12, F21, F12in, F21in