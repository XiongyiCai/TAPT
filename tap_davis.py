import time
import argparse
import numpy as np
import timeit
import cv2
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import glob
from PIL import Image
import mediapy as media
from typing import Iterable, Mapping, Tuple, Union
import pickle
from time import sleep



def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
  """Resize a video to output_size."""
  # If you have a GPU, consider replacing this with a GPU-enabled resize op,
  # such as a jitted jax.image.resize.  It will make things faster.
  return media.resize_video(video, output_size)

# path = './tapvid_davis/tapvid_davis.pkl'

# f = open(path, 'rb')

# data = pickle.load(f)

# dict_name = data.keys()

# data_to_gt = {} 

# resize_H = 480
# resize_W = 854

# for dict in dict_name:
    
#     S,H,W,C = data[dict]['video'].shape
#     n_frames = int(data[dict]['video'].shape[0])

    
#     vedio = resize_video(data[dict]['video'],(resize_H,resize_W)) # S ,H ,W ,C
    
    
#     target_point = data[dict]['points']
#     target_point *= np.array([
#         resize_H,
#         resize_W
#     ])  
    
#     target_point = target_point.transpose(1,0,2) # S , N ,2
    
    
#     occluded = data[dict]['occluded']
    
#     vis = 1-occluded.transpose(1,0) # S , N
    
    
    
#     data_to_gt[dict] = {
#         'rgb': np.copy(vedio),
#         'vis': np.copy(vis),
#         'traj': np.copy(target_point)
#     }




class DAVISDataset(torch.utils.data.Dataset):
    def __init__(self, seqlen=8,mode = 'seqlen'):
        path = '../tapvid_davis/tapvid_davis.pkl'
        f = open(path, 'rb')
        data = pickle.load(f)
        self.mode = mode
        self.dict_lens = []
        self.data_to_gt = {} 

        self.dict_name = list(data.keys())
        

        self.seqlen = seqlen

    
        resize_H = 256
        resize_W = 256
        
        for dict in self.dict_name:
            n_frames = int(data[dict]['video'].shape[0])
            self.dict_lens.append(n_frames // seqlen)
            


            
            video = resize_video(data[dict]['video'],(resize_H,resize_W)) # S ,H ,W, C
            
            
            target_point = data[dict]['points']
            
            
            target_point *= np.array([
                resize_W,
                resize_H
            ])  
            
            
            # target_point = target_point.transpose(1,0,2) # S , N ,2
            
            
            occluded = data[dict]['occluded']
            
            # vis =  1-occluded.transpose(1,0) # S , N
            

            
            self.data_to_gt[dict] = {
                'video': np.copy(video),
                'occluded': np.copy(occluded),
                'target_point': np.copy(target_point)
            }
            

    def __getitem__(self, index):
        # identify which sample and which starting frame it is
        subfolder_id = 0
        if self.mode == 'seqlen':
            while index >= self.dict_lens[subfolder_id]:
                index -= self.dict_lens[subfolder_id]
                subfolder_id += 1

            # start from subfolder and the frame
            subfolder = self.dict_name[subfolder_id]
            start_frame = index * self.seqlen

            # get gt
            S = self.seqlen
            video_s = self.data_to_gt[subfolder]['video'][start_frame:start_frame+S,:,:,:] # S ,H, W, C
            occ_s = self.data_to_gt[subfolder]['occluded'][:, start_frame:start_frame+S] # N, S
            traj_s = self.data_to_gt[subfolder]['target_point'][:, start_frame:start_frame+S,:] # N,S,2
            
        elif self.mode == 'clip':
            subfolder_id = index
            subfolder = self.dict_name[subfolder_id]
            video_s = self.data_to_gt[subfolder]['video'] # S ,H, W, C
            occ_s = self.data_to_gt[subfolder]['occluded'] # N, S
            traj_s = self.data_to_gt[subfolder]['target_point'] # N, S ,2
            subfolder_id += 1
        else:
            print('unknown mode')
        


        sample = {
            'video': video_s, # S ,H, W, C
            'occluded': occ_s, # N, S
            'target_points': traj_s, # N, S ,2
        }
        
        return sample

    def __len__(self):
        if self.mode == 'seqlen':     
            return sum(self.dict_lens)
        elif self.mode == 'clip':
            return len(self.dict_name)


