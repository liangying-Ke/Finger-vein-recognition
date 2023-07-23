import torch
import torch.nn as nn
from .diversebranchblock import DiverseBranchBlock

CONV_BN_IMPL = 'DBB'
DEPLOY_FLAG = False

def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return DiverseBranchBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG)

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return DiverseBranchBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG, nonlinear=nn.ReLU())

def switch_deploy_flag(deploy):
    global DEPLOY_FLAG
    DEPLOY_FLAG = deploy
    print('deploy flag: ', DEPLOY_FLAG)

