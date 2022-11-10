import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from hilbertcurve.hilbertcurve import HilbertCurve


class BasicKD(nn.Module):
    def __init__(self, temperature=4):
        super(BasicKD, self).__init__()
        self.T = temperature

    def forward(self, y, teacher_scores):
        p = F.log_softmax(y / self.T, dim=1)
        q = F.softmax(teacher_scores / self.T, dim=1)

        l_kl = F.kl_div(p, q, reduction='batchmean') * (self.T ** 2)
        return l_kl

def get_hilbert_feature(feature, n):
    side_length = max(feature.size(-1), feature.size(-2))

    # make the minimum entire hilbert space
    p = np.ceil(np.log2(side_length))
    hilbert_curve = HilbertCurve(p, n)
    distances = list(range(2 ** (p.astype(int) * n)))
    points = torch.tensor(hilbert_curve.points_from_distances(distances)).to(feature.device)

    if n == 3:
        points = points[(points[:, 0] < feature.size(2)) & (points[:, 1] < feature.size(3)) & (points[:, 2] < feature.size(4))]
        points = points[:, 0] * feature.size(2) * feature.size(3) + points[:, 1] * feature.size(3) + points[:, 2]
    else:
        points = points[(points[:, 0] < feature.size(2)) & (points[:, 1] < feature.size(3))]
        points = points[:, 0] * feature.size(2) + points[:, 1]

    feature = feature.view(feature.size(0), feature.size(1), -1)
    hf = feature.index_select(dim=-1, index=points)
    return hf


def hilbert_distillation(feat_s, feat_t):
    deform_t = nn.Conv3d(in_channels=feat_t.shape[1], out_channels=128, kernel_size=1).to(feat_t.device)  # 128
    deform_s = nn.Conv2d(in_channels=feat_s.shape[1], out_channels=128, kernel_size=1).to(feat_s.device)  # 128
    feat_t = deform_t(feat_t)
    feat_s = deform_s(feat_s)

    hf_t = get_hilbert_feature(feat_t, 3)
    hf_s = get_hilbert_feature(feat_s, 2)

    hf_t_norm = F.interpolate(hf_t, hf_s.size(2))

    # or use l2?
    # l_hd = F.kl_div(F.normalize(hf_s, dim=2), F.normalize(hf_t_norm, dim=2))
    # l_hd = F.kl_div(hf_s, hf_t_norm)
    l_hd = F.kl_div(torch.log(F.sigmoid(hf_s)), F.sigmoid(hf_t_norm))

    return l_hd


def variable_length_hilbert_distillation(feat_s, feat_t):
    # if feat_s.size(1) != feat_t.size(1):
    #     feat_t = feat_t.permute(0, 4, 2, 3, 1)
    #     feat_t = F.adaptive_avg_pool3d(feat_t, (feat_t.size(2), feat_t.size(3), feat_s.size(1)))
    #     feat_t = feat_t.permute(0, 4, 2, 3, 1)

    deform_t = nn.Conv3d(in_channels=feat_t.shape[1], out_channels=128, kernel_size=1).to(feat_t.device)  # 128
    deform_s = nn.Conv2d(in_channels=feat_s.shape[1], out_channels=128, kernel_size=1).to(feat_s.device)  # 128
    feat_t = deform_t(feat_t)
    feat_s = deform_s(feat_s)

    at_t = F.normalize(feat_t.pow(4).mean(1)).unsqueeze(1)
    at_s = F.normalize(feat_s.pow(4).mean(1)).unsqueeze(1)
    hf_t = get_hilbert_feature(feat_t, 3) * get_hilbert_feature(at_t, 3)
    hf_s = get_hilbert_feature(feat_s, 2) * get_hilbert_feature(at_s, 2)

    hf_t_norm = F.interpolate(hf_t, hf_s.size(2))

    # 对齐
    # find_key = nn.Linear(hf_s.size(2), hf_s.size(2)).to(feat_t.device)
    # hf_t_norm = find_key(hf_t_norm)

    # or use l2?

    # l_hd = F.kl_div(F.normalize(hf_s, dim=2), F.normalize(hf_t_norm, dim=2))
    l_hd = F.kl_div(hf_s, hf_t_norm)
    return l_hd