import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np
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

def basic_distillation(y, teacher_scores, T=4):
    """
    basic KD loss function
    :param y: student score
    :param teacher_scores: teacher score
    :param T: temperature
    :return: loss value
    """
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)

    l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
    return l_kl

def get_hilbert_feature(feature, n):
    side_length = max(feature.size(-1), feature.size(-2))

    # make the minimum filling hilbert space
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
    eps = 0.0000001

    deform_t = nn.Conv3d(in_channels=feat_t.shape[1], out_channels=128, kernel_size=1).to(feat_t.device)  # 128
    deform_s = nn.Conv2d(in_channels=feat_s.shape[1], out_channels=128, kernel_size=1).to(feat_s.device)  # 128
    feat_t = deform_t(feat_t)
    feat_s = deform_s(feat_s)

    hf_t = get_hilbert_feature(feat_t, 3)
    hf_s = get_hilbert_feature(feat_s, 2)

    hf_t = F.interpolate(hf_t, hf_s.size(2))

    hf_s_norm = torch.sqrt(torch.sum(hf_s ** 2, dim=1, keepdim=True))
    hf_s = hf_s / (hf_s_norm + eps)
    hf_s[hf_s != hf_s] = 0

    hf_t_norm = torch.sqrt(torch.sum(hf_t ** 2, dim=1, keepdim=True))
    hf_t = hf_t / (hf_t_norm + eps)
    hf_t[hf_t != hf_t] = 0

    l_hd = F.smooth_l1_loss(hf_s, hf_t)
    return l_hd


def variable_length_hilbert_distillation(feat_s, feat_t):
    eps = 0.0000001

    deform_t = nn.Conv3d(in_channels=feat_t.shape[1], out_channels=128, kernel_size=1).to(feat_t.device)  # 128
    deform_s = nn.Conv2d(in_channels=feat_s.shape[1], out_channels=128, kernel_size=1).to(feat_s.device)  # 128
    feat_t = deform_t(feat_t)
    feat_s = deform_s(feat_s)

    '''
    Inspired by Attention Transfer (https://arxiv.org/abs/1612.03928),
    here we adopt a more efficient approximation approach to achieve the variable-length by calculating 
    Activation Mapping (AM) through attention map
    '''
    at_t = F.normalize(feat_t.pow(4).mean(1)).unsqueeze(1)
    at_s = F.normalize(feat_s.pow(4).mean(1)).unsqueeze(1)
    hf_t = get_hilbert_feature(feat_t, 3) * get_hilbert_feature(at_t, 3)
    hf_s = get_hilbert_feature(feat_s, 2) * get_hilbert_feature(at_s, 2)

    hf_t = F.interpolate(hf_t, hf_s.size(2))

    hf_s_norm = torch.sqrt(torch.sum(hf_s ** 2, dim=1, keepdim=True))
    hf_s = hf_s / (hf_s_norm + eps)
    hf_s[hf_s != hf_s] = 0

    hf_t_norm = torch.sqrt(torch.sum(hf_t ** 2, dim=1, keepdim=True))
    hf_t = hf_t / (hf_t_norm + eps)
    hf_t[hf_t != hf_t] = 0

    l_hd = F.smooth_l1_loss(hf_s, hf_t)

    return l_hd


def diffusion_kd_loss(feat_s, feat_t, sample_index, sample_depth, T=4):
    """
    diffusion KD loss function
    :param T: Temperature
    :param feat_s: student feature (N,C,H,W)
    :param feat_t: teacher feature (N,C,D,H,W)
    :param index: the index number of student input among teacher input along the depth dimension
    :param index: the depth of teacher input
    :return: loss value
    """
    feat_t = feat_t.permute(0, 1, 3, 4, 2)
    depth = feat_t.shape[-1]
    index = sample_index // (sample_depth // depth)
    dkd_loss = 0
    k = 1
    while True:
        d_min = max(0, index-k)
        d_max = min(depth, index+k+1)
        feat_t_temp = feat_t[..., d_min:d_max]

        sm_t = channel_wise_sim_map(feat_t_temp)
        sm_s = channel_wise_sim_map(feat_s)

        p = F.log_softmax(sm_s / T, dim=-2)
        q = F.softmax(sm_t / T, dim=-2)
        l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)

        dkd_loss += math.exp(1-k) * l_kl
        # print(f'k={k}, l_kl: {l_kl}, dkd：{dkd_loss}')
        if k * 2 >= depth:
            break
        k = k * 2

    return dkd_loss


def channel_wise_sim_map(x):
    """
    Compute similarity map of a feature map
    :param x: feature
    :return: similarity map
    """
    x = x.reshape(x.size(0), x.size(1), -1)    # N,C,H,W => N,C,H*W or N,C,H,W,D => N,C,H*W*D
    col = x.unsqueeze(-2)
    row = x.unsqueeze(-3)
    col, row = torch.broadcast_tensors(col, row)
    cos = nn.CosineSimilarity(dim=-1)
    sm = cos(col, row)
    return sm

