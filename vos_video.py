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

class Vos_video(Dataset):
    def __init__(self, JPEGPath, AnnPath, json_path, seqlen, height, width):
        
        self.JPEGPath = JPEGPath
        self.AnnPath = AnnPath
        self.json_path = json_path

        self.folders = os.listdir(self.JPEGPath)
        json_file = open(self.json_path)
        json_str = json_file.read()
        self.json_data = json.loads(json_str)
        self.seqlen = seqlen
        
        self.H = height
        self.W = width

    def __getitem__(self,index):
        
        selectFolder = self.folders[index]
        
        folder_list_all = os.listdir(self.JPEGPath + '/' + selectFolder)
        
        ann_folder_list = os.listdir(self.AnnPath + '/' + selectFolder)
        
        ann_id_list = []
        
        ann_list = []
        
        folder_list_all.sort(key=lambda x: float(x.split(".")[0]))
        
        total_frame = len(folder_list_all)
        
        while len(ann_list) == 0:
            try:
                start = random.randint(0, total_frame - self.seqlen-1)
            except:
                print(selectFolder)
                print(1)
                return torch.ones(1), False
            
            folder_list = folder_list_all[start : start + self.seqlen]

            rgb_list = []

            for i in range(self.seqlen):
                
                
                if ((folder_list[0].split(".")[0] + '.png') in ann_folder_list) and (i == 0):
                    annpth = self.AnnPath + '/' + selectFolder + '/' + folder_list[0].split(".")[0] + '.png'
                    try:
                        mask =  np.array(Image.open(annpth)) # H, W
                        ann_id_list.append(int(i))
                        ann_list.append(mask)
                    except:
                        print(selectFolder)
                        print(3)
                        return torch.ones(1), False
                    
                if len(ann_list) != 0:
                    rgbpth = self.JPEGPath + '/' + selectFolder + '/' + folder_list[i]
                    
                    try:
                        img =  np.array(Image.open(rgbpth)) # H, W, 3
                    except:
                        print(selectFolder)
                        print(2)
                        return torch.ones(1), False
                    
                        # cv2.imwrite('pic/a.jpg', img*mask[:,:,None].astype(bool))
                    
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
                'id': ann_id_list,
                }, True
        
        
            
    def __len__(self): 
        return len(self.folders)

            
            

if __name__ == '__main__':
    
    train_dataset = Vos_video(JPEGPath = '../vos/train_all_frames_zip/train_all_frames/JPEGImages', 
                                AnnPath = '../vos/train/Annotations', 
                                json_path = '../vos/train/meta.json', seqlen = 48, height = 384, width = 512)
    
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
        # if gotit == False:
        #     print(1)
        a = 1
        
        
            