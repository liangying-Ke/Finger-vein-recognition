import os
import torch
import shutil

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(filename, state, is_bestF1, is_bestAcc, is_bestLoss):
    torch.save(state, filename)

    if is_bestF1 or is_bestAcc or is_bestLoss:
        print('Saving best model...')
    if is_bestF1:
        shutil.copyfile(filename, filename.replace('pth.tar', 'bestF1.pth.tar'))
    if is_bestAcc:
        shutil.copyfile(filename, filename.replace('pth.tar', 'bestAcc.pth.tar'))
    if is_bestLoss:
        shutil.copyfile(filename, filename.replace('pth.tar', 'bestLoss.pth.tar'))
