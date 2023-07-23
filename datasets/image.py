# 匯入基本的套件
import cv2
import torch
import pickle
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

class ImagesDataset(Dataset):
    # 呼叫此類別時所要進行的初始化
    def __init__(self, args, data_type, phase='train'):
        self.args = args
        self.phase = phase
        self.data_type = data_type
        self._read_path_label()
        self._setup_transformation(self.phase)
        self._get_label_list()

    def _read_path_label(self):
        pkl = pickle.load(open(self.args.annot_file, 'rb'))

        if self.data_type is not None:
            pkl = pkl[self.data_type]
        if self.phase == 'train':
            self.data = pkl['Training_Set']
        elif self.phase == 'val':
            self.data = pkl['Validating_Set']
        elif self.phase == 'test':
            self.data = pkl['Testing_Set']
        else:
            raise ValueError("train mode must be in : Train or Validation")
        self.dataset_size = len(self.data)
        self._get_mean_std(pkl['Training_Set'])

    def _get_mean_std(self, data):
        dataset_size = len(data)
        self.mean = np.zeros(1)
        self.std = np.zeros(1)
        for idx in range(dataset_size):
            img = cv2.imread(data[idx]['path'], flags=0)
            self.mean += np.mean(img)
            self.std += np.std(img)
        self.mean = self.mean / dataset_size / 255.
        self.std = self.std / dataset_size / 255.
        self.mean = self.mean.repeat(3)
        self.std = self.std.repeat(3)
            
    def _get_label_list(self):
        self.label_list = [data['label'] for data in self.data]
    
    def _setup_transformation(self, phase):
        self.phase = phase
        if self.phase == 'train':
            self.transforms = A.Compose([
                A.PadIfNeeded(min_height=self.args.pad_height_width, min_width=self.args.pad_height_width, border_mode=0),
                A.Resize(self.args.img_size, self.args.img_size), 
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])

        elif self.phase == 'test' or self.phase == 'val':
            self.transforms = A.Compose([
                A.PadIfNeeded(min_height=self.args.pad_height_width, min_width=self.args.pad_height_width, border_mode=0),
                A.Resize(self.args.img_size, self.args.img_size), 
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx]['path'])
        img = self.transforms(image=img)['image'] 
        label = self.data[idx]['label']
        label = torch.tensor(label, dtype=torch.long)
        return img, label


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_image, self.next_label = next(self.loader)
        except StopIteration:
            self.next_image = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_image = self.next_image.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        image = self.next_image
        label = self.next_label 
        self.preload()
        return image, label
