import torch
import numpy as np
import os
import scipy.io as scio
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import random 


def readTXT(self, txt_path):
    with open(txt_path, 'r') as f:
        listInTXT = [line.strip() for line in f]

    return listInTXT


class Nyu2Dataset(Dataset):
    def __init__(self, data_path: str, transform=None, istrain=True):
        self.data_path = data_path
        
        if istrain:
            self.txt_path = os.path.join(self.data_path, 'train_list.txt')
            self.image_path = os.path.join(self.data_path, 'sync')
            self.depth_path = os.path.join(self.data_path, 'sync')
        else:
            self.txt_path = os.path.join(self.data_path, 'test_list.txt')
            self.image_path = os.path.join(self.data_path, 'official_splits/test')
            self.depth_path = os.path.join(self.data_path, 'official_splits/test')
        self.filenames = readTXT(self, self.txt_path)

        self.transform = transform

    def __getitem__(self, idx: int) -> dict:
        filename = self.filenames[idx].split(' ')
        image = Image.open(self.image_path + filename[0])
        depth = Image.open(self.depth_path + filename[1])

        image = np.array(image)
        depth = np.array(depth)
        depth = depth / 1000.0
        depth = depth[..., np.newaxis]
        additional_targets = {'depth': 'mask'}
        aug = A.Compose(transforms=self.transform,
                        additional_targets=additional_targets)
        sample1 = aug(image=image, depth=depth)

            
        return sample1['image'].float()/255.0, sample1['depth'].float()

    def __len__(self):
        return len(self.filenames)


class SQUIDdataset(Dataset):
    def __init__(self, data_path: str, transform=None, is_predict=False, istrain=True): 
        self.data_path = data_path
        self.transform = transform
        self.is_predict = is_predict
        self.image_path = os.path.join(self.data_path, 'undistorted_images')
        self.depth_path = os.path.join(self.data_path, 'depth_mats')

        if is_predict:
            self.filenames = os.listdir(self.image_path)
        else:
            if istrain:
                self.txt_path = os.path.join(self.data_path, 'train_list.txt')
            else:
                self.txt_path = os.path.join(self.data_path, 'test_list.txt')
            self.filenames = readTXT(self, self.txt_path)
            

    def __getitem__(self, idx: int) -> dict:
        filename = self.filenames[idx]
        image = Image.open(os.path.join(self.image_path, filename))
        if filename.split('_')[-2]=='LFT':
            mat = scio.loadmat(os.path.join(self.depth_path, filename.split('LFT')[0]+'depth.mat'))
            depth = mat['dist_map_l']
        else:
            mat = scio.loadmat(os.path.join(self.depth_path, filename.split('RGT')[0]+'depth.mat'))
            depth = mat['dist_map_r']
        depth[np.isnan(depth)] = -1
        depth = depth[..., np.newaxis]
        image = np.array(image)
        depth = np.array(depth)
        
        # if self.is_predict:
        #     resize_h = image.shape[0]//32 // 2 * 32
        #     resize_w = image.shape[1]//32 // 2 * 32
        #     image = cv2.resize(image, (resize_w, resize_h))
        #     depth = cv2.resize(depth, (resize_w, resize_h))
        #     image = self.transform(image)
        #     depth = self.transform(depth)
        # else:
        additional_targets = {'depth': 'mask'}
        aug = A.Compose(transforms=self.transform,
                        additional_targets=additional_targets)
        sample1 = aug(image=image, depth=depth)
        image = sample1['image']
        depth = sample1['depth']
        return image.float()/255.0, depth.float()
        

    def __len__(self):
        return len(self.filenames)

class MyDataModule(LightningDataModule):
    def __init__(self, data_name, data_path, predict_data_path, scale_size, FDA_trans = False, batch_size=4, numworkers=0):
        super().__init__()
        self.data_name = data_name
        self.data_path = data_path
        self.predict_data_path = predict_data_path
        self.scale_size = scale_size
        self.batch_size = batch_size
        self.numworkers = numworkers

        if FDA_trans:
            reference_images = glob.glob(predict_data_path + '/undistorted_images/*.png')
            self.train_transform = [
                A.Resize(self.scale_size[0], self.scale_size[1]),
                A.FDA(reference_images=reference_images, beta_limit=0.01, p=1.0),
                A.Blur(blur_limit=3, p=0.5),
                ToTensorV2(transpose_mask=True)
            ]
        else:
            self.train_transform = [
                A.Resize(self.scale_size[0], self.scale_size[1]),
                A.HorizontalFlip(),
                # A.RandomCrop(crop_size[0], crop_size[1]),
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
                A.HueSaturationValue(),
                ToTensorV2(transpose_mask=True)
            ]  
        self.test_transform = [
            A.Resize(self.scale_size[0], self.scale_size[1]),
            ToTensorV2(transpose_mask=True)
        ]


    def train_dataloader(self):
        if self.data_name=='nyu2':
            train_dataset = Nyu2Dataset(self.data_path, transform=self.train_transform , istrain=True)
        elif self.data_name=='SQUID':
            train_dataset = SQUIDdataset(self.data_path, transform=self.train_transform , istrain=True)
        else:
            raise ValueError('data Error!')
        train_loader = DataLoader(
            train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.numworkers,
            pin_memory=False)

        return train_loader

    def val_dataloader(self):
        if self.data_name == 'nyu2':
            test_dataset = Nyu2Dataset(self.data_path, transform=self.test_transform, istrain=False)
        elif self.data_name == 'SQUID':
            test_dataset = SQUIDdataset(self.data_path, transform=self.test_transform, istrain=False)
        else:
            raise ValueError('data Error!')
        test_loader = DataLoader(
            test_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.numworkers,
            pin_memory=False)

        return test_loader

    def predict_dataloader(self):
        predict_dataset = SQUIDdataset(self.predict_data_path, self.test_transform, is_predict=True)
        predict_loader = DataLoader(
            predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.numworkers,
            pin_memory=False)

        return predict_loader



