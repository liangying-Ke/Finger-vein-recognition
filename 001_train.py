import warnings
warnings.filterwarnings("ignore")

import os
import torch
import models 
import datasets
import argparse
import torch.nn as nn
import sklearn.metrics as skm 
import torch.nn.functional as F

from utils import *
from tqdm import tqdm
from torch.nn import init
from pytorch_model_summary import summary


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default="PLUSVein")
    parser.add_argument('--optim', type=str, default="adamw")
    parser.add_argument('--scheduler', type=str, default='Cosine')
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    return parser.parse_args([])


def getDatasetParams(args):
    if args.datasets == 'FVUSM':
        args.classes = 492
        args.pad_height_width = 300
        args.data_type = [None]
        args.init_type = 'xavier_normal'
        args.root_model = './checkpoint/FV-USM'
        args.annot_file = './datasets/annotations_fvusm.pkl'
    elif args.datasets == 'PLUSVein':
        args.classes = 360
        args.pad_height_width = 736
        args.data_type = ['LED', 'LASER']
        args.init_type = 'xavier_uniform'
        args.root_model = './checkpoint/PLUSV-FV3'
        args.annot_file = './datasets/annotations_plusvein.pkl'
    return args

def getOptimParams(args):
    if args.optim == 'adamw':
        args.lr = 2e-4
        args.weight_decay = 1e-4
    if args.optim == 'sgd':
        args.lr = 1e-1
        args.momentum = 0.9
        args.weight_decay = 2e-4
        
    if args.scheduler == 'Cosine':
        args.T_max = 32
        args.eta_min = 1e-6
    if args.scheduler == 'ReduceLROnPlateau':
        args.factor = 0.9
        args.patience = 10
        args.verbose = True
    return args


def _get_optimizer(args, model):
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.T_max, eta_min=args.eta_min)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=args.factor, patience=args.patience, verbose=args.verbose)
    else:
        scheduler = None
    return optimizer, scheduler


def init_weights(net, init_type='normal', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)


def _get_model(args):
    models.switch_deploy_flag(False)
    model = models.LightWeightedModel(num_classes=args.classes).to(args.device)
    elasticFace = models.ElasticArcFace().to(args.device)
    init_weights(model, args.init_type)
    return model, elasticFace


def Learning(*kwargs):
    all_preds = []
    all_targets = []
    losses = AverageMeter('Loss', ':.4e')

    args = kwargs[0]
    model = kwargs[1]
    elasticFace = kwargs[2]
    DataLoader = kwargs[3]
    criterion_cls = kwargs[4]
    optimizer = kwargs[5]
    train_infos = kwargs[-1] 

    model.train() if args.phase == "train" else model.eval()
    for _ in tqdm(range(args.max_iteration_train if args.phase == 'train' else args.max_iteration_val)):
        inputs, targets = DataLoader.next()
        if inputs is None or targets is None:
            break
        with torch.set_grad_enabled(args.phase=="train"):
            cos_theta, outputs  = model(inputs)
            cos_theta = elasticFace(cos_theta, targets)
            loss = criterion_cls(cos_theta, targets)
            if args.phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        outputs = F.sigmoid(outputs)
        _, pred = torch.max(outputs, 1)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        losses.update(loss.item(), inputs.size(0))

    acc = skm.accuracy_score(all_targets, all_preds).astype(float)*100
    f1 = skm.f1_score(all_targets, all_preds, average='macro').astype(float)*100

    if args.phase == "train": 
        train_infos += f"\nTrain Loss: {losses.avg:.3f}"
        train_infos += f"\nTrain Accuracy: {acc:.2f}"
        return args, model, train_infos
    val_infos = f'Val Loss: {losses.avg:.4f}'
    val_infos += f'\nVal Accuracy: {acc:.2f}'
    val_infos += f"\nVal F1-Score: {f1:.2f}"
    print('-'*100)
    print(train_infos)
    print('-'*100)
    print(val_infos)
    print('-'*100)
    return losses.avg, acc, f1


def main(args):
    for data_type in args.data_type:
        train_dataset = datasets.ImagesDataset(args=args, data_type=data_type, phase='train')
        val_dataset = datasets.ImagesDataset(args=args, data_type=data_type, phase='val')
        Train_DataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, persistent_workers=True, pin_memory=True, drop_last=True)
        Val_DataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, persistent_workers=True, pin_memory=True)
        args.max_iteration_train = len(Train_DataLoader)
        args.max_iteration_val = len(Val_DataLoader)

        best_loss = float('inf')
        best_acc = float('-inf')
        best_f1 = float('-inf')

        model, elasticFace = _get_model(args)
        criterion = nn.CrossEntropyLoss().to(args.device)
        optimizer, scheduler = _get_optimizer(args, model)

        input1 = torch.zeros([1, 3, args.img_size, args.img_size], device=args.device)
        print(summary(model, input1, show_input=False))
        print(args)
        loss_history = []
        for epoch in range(args.epochs):
            args.epoch = epoch + 1
            TrainDataLoaderPrefetcher = datasets.data_prefetcher(Train_DataLoader)
            ValDataLoaderPrefetcher = datasets.data_prefetcher(Val_DataLoader)

            train_infos = f"Epoch: [{args.epoch:03d}/{args.epochs:03d}]"
            args.phase = 'train'
            args, model, train_infos = Learning(args, model, elasticFace, TrainDataLoaderPrefetcher, criterion, optimizer, train_infos)
            args.phase = 'val'
            loss, acc, f1 = Learning(args, model, elasticFace, ValDataLoaderPrefetcher, criterion, optimizer, train_infos)
            loss_history.append(loss)

            if args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(loss)
            elif args.scheduler != '':
                scheduler.step()

            is_bestLoss = loss < best_loss
            is_bestF1 = f1 > best_f1
            is_bestAcc = acc > best_acc
            best_loss = min(loss, best_loss)
            best_f1 = max(f1, best_f1)
            best_acc = max(acc, best_acc)

            save_path = os.path.join(args.root_model, str(data_type)) 
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            save_checkpoint(os.path.join(save_path, f"Backbone_ckpt.pth.tar"), {
                'args': args,
                'Accuracy': acc,
                'F1-Score': f1,
                'model_state_dict': model.state_dict(),
            }, is_bestF1, is_bestAcc, is_bestLoss)
            
            print(f'Best Loss: {best_loss:.6f}')
            print(f'Best F1-Score: {best_f1:.2f}')
            print(f'Best Accuracy: {best_acc:.2f}')
            print('='*100)


if __name__ == '__main__':
    args = get_argument()
    args = getOptimParams(args)
    args = getDatasetParams(args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
