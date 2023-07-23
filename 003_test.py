import warnings
warnings.filterwarnings("ignore")

import os
import time
import torch
import models 
import datasets
import argparse
import torch.nn.functional as F

from utils import *
from tqdm import tqdm
from pytorch_model_summary import summary
from ptflops import get_model_complexity_info

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default="PLUSVein")
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    return parser.parse_args([])

def getDatasetParams(args):
    if args.datasets == 'FVUSM':
        args.classes = 492
        args.pad_height_width = 300
        args.data_type = [None]
        args.root_model = './checkpoint/FV-USM'
        args.annot_file = './datasets/annotations_fvusm.pkl'
    elif args.datasets == 'PLUSVein':
        args.classes = 360
        args.pad_height_width = 736
        args.data_type = ['LED', 'LASER']
        args.root_model = './checkpoint/PLUSV-FV3'
        args.annot_file = './datasets/annotations_plusvein.pkl'
    return args

def _get_model(args):
    models.switch_deploy_flag(True)
    model = models.LightWeightedModel(num_classes=args.classes).to(args.device)
    return model

def test(*kwargs):
    # 紀錄預測結果、目標標籤
    args = kwargs[0]
    model = kwargs[1]
    DataLoader = kwargs[-1]
    count = 0
    correct = 0
    model.eval()
    start = time.time()
    for inputs, targets in tqdm(DataLoader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        with torch.set_grad_enabled(False):
            _, outputs  = model(inputs)
            outputs = F.sigmoid(outputs)
            confidence, pred = torch.max(outputs, dim=1)
            for idx, conf in enumerate(confidence):
                if conf >= 0.5 and pred[idx] == targets[idx]:
                    correct += 1
                count += 1
    cost_time = (time.time() - start) / count
    acc = correct / count * 100
    return acc, cost_time

def main(args):
    for data_type in args.data_type:
        test_dataset = datasets.ImagesDataset(args=args, data_type=data_type, phase='test')
        model = _get_model(args)
        input1 = torch.zeros([1, 3, args.img_size, args.img_size], device=args.device)
        print(summary(model, input1, show_input=False))
        macs, _ = get_model_complexity_info(model, (3, args.img_size, args.img_size), as_strings=True, print_per_layer_stat=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('='*100)
        for metrics in ['F1', 'Acc', 'Loss']:
            path = os.path.join(args.root_model, data_type) if data_type is not None else args.root_model
            weights = torch.load(os.path.join(path, 'switchDBB', f"Backbone_ckpt.best{metrics}.pth.tar"))
            Test_DataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            model.load_state_dict(weights['model_state_dict']) 
            acc, cost_time = test(args, model, Test_DataLoader)
            print(f'data_type: {data_type}, Metrics: {metrics}')
            print(f'Accuracy: {acc:.2f}')
            print(f'Cost_times: {cost_time*1000} ms')
            print('-'*100)
        print('='*100)

if __name__ == '__main__':
    args = get_argument()
    args = getDatasetParams(args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
