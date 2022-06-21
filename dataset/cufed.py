import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import random
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR'] = np.rot90(sample['LR'], k1).copy()
        sample['HR'] = np.rot90(sample['HR'], k1).copy()
        sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.fliplr(sample['LR']).copy()
            sample['HR'] = np.fliplr(sample['HR']).copy()
            sample['LR_sr'] = np.fliplr(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.fliplr(sample['Ref']).copy()
            sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
            sample['LR_sr'] = np.flipud(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.flipud(sample['Ref']).copy()
            sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        LR, LR_sr, HR, Ref, Ref_sr = sample['LR'], sample['LR_sr'], sample['HR'], sample['Ref'], sample['Ref_sr']
        # print(LR.shape)
        # print(LR_sr.shape)
        # exit(0)
        LR = LR.transpose((2,0,1))
        LR_sr = LR_sr.transpose((2,0,1))
        HR = HR.transpose((2,0,1))
        Ref = Ref.transpose((2,0,1))
        Ref_sr = Ref_sr.transpose((2,0,1))
        return {'LR': torch.from_numpy(LR).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float()}

class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()]) ):
        self.input_list = sorted([os.path.join(args.dataset_dir, 'train/input', name) for name in
            os.listdir( os.path.join(args.dataset_dir, 'train/input') )])
        self.ref_list = sorted([os.path.join(args.dataset_dir, 'train/ref', name) for name in
            os.listdir( os.path.join(args.dataset_dir, 'train/ref') )])
        self.transform = transform
        self.crop_size = 160

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        # HR = imread(self.input_list[idx])
        HR = cv.imread(self.input_list[idx],0)

        # 得到 i，j, th, tw
        # 得到 cropsize
        h, w = HR.shape[:2]
        th, tw = self.crop_size, self.crop_size
        i = 0
        j = 0
        if th != h and tw != w:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

        # 对 HR 进行裁剪
        HR = HR[i : i + th,j : j + tw]
        #HR = HR[:h//4*4, :w//4*4, :]

        ### LR and LR_sr
        # Image.fromarray(HR) 实现 array 到 image 的转换
        LR = np.array(Image.fromarray(HR).resize((tw//4, th//4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((tw, th), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref_sub = cv.imread(self.ref_list[idx],0)

        # 对 Ref 进行裁剪
        Ref_sub = Ref_sub[i: i + th, j: j + tw]

        h2, w2 = Ref_sub.shape[:2]
        Ref_sr_sub = np.array(Image.fromarray(Ref_sub).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr_sub = np.array(Image.fromarray(Ref_sr_sub).resize((w2, h2), Image.BICUBIC))

        ### complete ref and ref_sr to the same size, to use batch_size > 1
        # # ?
        Ref = np.zeros((160, 160))
        Ref_sr = np.zeros((160, 160))
        Ref[:h2, :w2] = Ref_sub
        Ref_sr[:h2, :w2] = Ref_sr_sub

        # Ref = Ref_sub
        # Ref_sr = Ref_sr_sub

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        LR = np.expand_dims(LR,axis=-1).repeat(3,axis=2)
        LR_sr = np.expand_dims(LR_sr,axis=-1).repeat(3,axis=2)
        HR = np.expand_dims(HR,axis=-1).repeat(3,axis=2)
        Ref = np.expand_dims(Ref, axis=-1).repeat(3, axis=2)
        Ref_sr = np.expand_dims(Ref_sr, axis=-1).repeat(3, axis=2)
        # print(type(Ref_sr))
        # exit(0)
        sample = {'LR': LR,
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref,
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        # print(sample['LR'].shape)
        # exit(0)
        return sample


class TestSet(Dataset):
    def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()])):
        # self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5', '*_0.jpeg')))
        # self.ref_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5',
        #     '*_' + ref_level + '.jpg')))
        # eval
        self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'eval/CUFED5', '*_0.jpeg')))
        self.ref_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'eval/CUFED5',
                                                      '*_' + ref_level + '.jpg')))
        # print(os.path.join(args.dataset_dir, 'eval/CUFED5', '*_0.jpeg'))
        # exit(0)
        self.transform = transform
        # print(self.input_list)
        # print(os.path.join(args.dataset_dir, 'test/CUFED5', '*_0.jpg'))
        # exit(0)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = cv.imread(self.input_list[idx],0)
        h, w = HR.shape[:2]
        h, w = h//4*4, w//4*4
        HR = HR[:h, :w] ### crop to the multiple of 4

        ### LR and LR_sr
        LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref = cv.imread(self.ref_list[idx],0)
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2//4*4, w2//4*4
        Ref = Ref[:h2, :w2]
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        # print(LR)
        # exit(0)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.
        #
        LR = np.expand_dims(LR, axis=-1).repeat(3, axis=2)
        LR_sr = np.expand_dims(LR_sr, axis=-1).repeat(3, axis=2)
        HR = np.expand_dims(HR, axis=-1).repeat(3, axis=2)
        Ref = np.expand_dims(Ref, axis=-1).repeat(3, axis=2)
        Ref_sr = np.expand_dims(Ref_sr, axis=-1).repeat(3, axis=2)
        # print(LR.shape)
        # print(LR_sr.shape)
        # print(HR.shape)
        # exit(0)

        sample = {'LR': LR,  
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return sample