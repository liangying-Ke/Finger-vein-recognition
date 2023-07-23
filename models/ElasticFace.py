import torch
import torch.nn as nn
import torch.nn.functional as F


class ElasticArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.50, std=0.0125):
        super().__init__()
        # 設定縮放因子的值
        self.s = s
        # 設定常態分佈的平均值
        self.m = m
        # 設定常態分佈的標準差
        self.std=std
        
    def forward(self, cos_theta, label):
        # 取出所有標籤中標籤不為-1的索引位置
        index = torch.where(label != -1)[0]
        # 根據標籤進行one-hot向量的轉換
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        # 取得常態分佈的值
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device)
        # 取得margin的one-hot形式
        m_hot.scatter_(1, label[index, None], margin)
        # 使用arccos來取得角度值
        cos_theta.acos_()
        # 將角度加上margin
        cos_theta[index] += m_hot
        # 求得餘弦值，並且在後續乘上縮放因子(s)
        cos_theta.cos_().mul_(self.s)
        # 回傳Target Logit
        return cos_theta