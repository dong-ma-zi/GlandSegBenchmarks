import torch
import torch.nn as nn
import torch.nn.functional as F

class PairConsistency(nn.Module):
    def __init__(self, t):
        super(PairConsistency, self).__init__()
        self.t = t

    def forward(self, feats):
        b, c, h, w = feats.size()
        attention_maps = []
        feat0 = torch.permute(feats[0], (2, 0, 1))
        feat0 = torch.reshape(feat0, (c, h*w))
        for i in range(1, b):
            feati = torch.reshape(feats[i], (h*w, c))
            relaMatrix = F.softmat(torch.mm(feat0, feati), dim=0)
            attnMap = torch.mm(feati, relaMatrix)
            attnMap = torch.reshape(attnMap, (c, h, w))
            attention_maps.append(attnMap)

        v = torch.zeros((c, h, w)).cuda()
        D = len(attention_maps)
        for i in range(D):
            for j in range(i+1, D):
                v += torch.abs(attention_maps[i]-attention_maps[j])

        M = torch.zeros((c, h, w)).cuda()
        consistencyMap = torch.lt(v, self.t)

        for i in range(D):
            M += consistencyMap * attention_maps[i]
        M = M / D
        return M + feat0





