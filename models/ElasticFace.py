import torch
import torch.nn as nn
import torch.nn.functional as F


class ElasticArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.50, std=0.0125):
        super().__init__()
        self.s = s
        self.m = m
        self.std=std
        
    def forward(self, cos_theta, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], margin)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta
