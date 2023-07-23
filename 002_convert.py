import os
import torch
import models 
import argparse

# 設定訓練參數
def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default="PLUSVein")
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
    models.switch_deploy_flag(False)
    model = models.LightWeightedModel(num_classes=args.classes).to(args.device)
    return model

def convert(args):
    for data_type in args.data_type:
        for metrics in ['F1', 'Acc', 'Loss']:
            print(f'Data type: {data_type}, metrics: {metrics}')
            model = _get_model(args)
            path = os.path.join(args.root_model, data_type) if data_type is not None else args.root_model
            weights = torch.load(os.path.join(path, f"Backbone_ckpt.best{metrics}.pth.tar"))
            model.load_state_dict(weights['model_state_dict']) 
            for m in model.modules():
                if hasattr(m, 'switch_to_deploy'):
                    m.switch_to_deploy()
            weights['model_state_dict'] = model.state_dict()
            save_path = os.path.join(path, 'switchDBB')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(weights, os.path.join(save_path, f"Backbone_ckpt.best{metrics}.pth.tar"))

if __name__ == '__main__':
    args = get_argument()
    args = getDatasetParams(args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    convert(args)