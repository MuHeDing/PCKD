import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Correlation(nn.Module):

    def __init__(self, gamma=0.4, P_order=2):
        super(Correlation, self).__init__()
        self.gamma = gamma
        self.P_order = P_order

    def forward(self, feat_s, feat_t):
        corr_mat_s = self.get_correlation_matrix(feat_s)
        corr_mat_t = self.get_correlation_matrix(feat_t)

        loss = F.mse_loss(corr_mat_s, corr_mat_t)

        return loss

    def get_correlation_matrix(self, feat):
        feat = F.normalize(feat, p=2, dim=-1)
        sim_mat  = torch.matmul(feat, feat.t())
        corr_mat = torch.zeros_like(sim_mat)

        for p in range(self.P_order+1):
            corr_mat += math.exp(-2*self.gamma) * (2*self.gamma)**p / \
                math.factorial(p) * torch.pow(sim_mat, p)
        
        return corr_mat

# class Correlation(nn.Module):
#     """Similarity-preserving loss. My origianl own reimplementation 
#     based on the paper before emailing the original authors."""
#     def __init__(self):
#         super(Correlation, self).__init__()
#
#     def forward(self, f_s, f_t):
#         return self.similarity_loss(f_s, f_t)
#         # return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
#
#     def similarity_loss(self, f_s, f_t):
#         bsz = f_s.shape[0]
#         f_s = f_s.view(bsz, -1)
#         f_t = f_t.view(bsz, -1)
#
#         G_s = torch.mm(f_s, torch.t(f_s))
#         G_s = G_s / G_s.norm(2)
#         G_t = torch.mm(f_t, torch.t(f_t))
#         G_t = G_t / G_t.norm(2)
#
#         G_diff = G_t - G_s
#         loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
#         return loss
