import os
import sys
import numpy as np
import cv2
import json
import random
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import glob
import math
from tqdm import tqdm
import mediapy as media
import time

class Davis_video(Dataset):
    def __init__(self, JPEGPath, AnnPath, height, width):
        
        self.JPEGPath = JPEGPath
        self.AnnPath = AnnPath

        self.folders = os.listdir(self.JPEGPath)
  
        self.H = height
        self.W = width

    def __getitem__(self,index):
        
        selectFolder = self.folders[index]
        
        folder_list_all = os.listdir(self.JPEGPath + '/' + selectFolder)
        
        folder_list_all.sort(key=lambda x: float(x.split(".")[0]))
        
        ann_folder_list = os.listdir(self.AnnPath + '/' + selectFolder)
        
        ann_folder_list.sort(key=lambda x: float(x.split(".")[0]))
        
        rgb_list = []
        
        ann_list = []
        
        
        total_frame = len(folder_list_all)

        for i in range(total_frame):
            
            

            annpth = self.AnnPath + '/' + selectFolder + '/' + ann_folder_list[i]
            
            
            
            try:
                mask =  np.array(Image.open(annpth)) # H, W
                ann_list.append(mask)
            except:
                print(selectFolder)
                print(3)
                return torch.ones(1), False
                

            rgbpth = self.JPEGPath + '/' + selectFolder + '/' + folder_list_all[i]
            
            try:
                img =  np.array(Image.open(rgbpth)) # H, W, 3
            except:
                print(selectFolder)
                print(2)
                return torch.ones(1), False
            
            rgb_list.append(img)
            
        video = np.stack(rgb_list, axis = 0)
        
        mask = np.stack(ann_list, axis = 0)
        mask = mask[:,:,:,None]
        mask = torch.from_numpy(mask)
        mask = mask.permute(0,3,1,2)
        mask = F.interpolate(mask, size=(self.H, self.W), mode='nearest')
        
        mask = mask.permute(0,2,3,1)
        mask = mask.numpy()
        
        video = media.resize_video(video, (self.H, self.W))      
        
        return {'video': video,
                'mask': mask,
                }, True
        
        
            
    def __len__(self): 
        return len(self.folders)

            
            

if __name__ == '__main__':
    
    train_dataset = Davis_video(JPEGPath = '/GPFS/data/xiongyicai/BADJA/DAVIS/JPEGImages/Full-Resolution', 
                                AnnPath = '/GPFS/data/xiongyicai/BADJA/DAVIS/Annotations/Full-Resolution', 
                                height = 384, width = 512)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        drop_last = True,
        pin_memory=True,
        )
    

    i = 0
    for data, gotit in tqdm(train_dataloader):
        print(data['mask'].shape)
        
        
            