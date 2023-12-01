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



class RGBDataset(torch.utils.data.Dataset):
    def __init__(self, seqlen=8,mode = 'clip'):
        path = '/GPFS/data/xiongyicai/vid-data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl'
        f = open(path, 'rb')
        data = pickle.load(f)
        self.mode = mode
        self.dict_lens = []
        self.data_to_gt = [] 

        self.dict_num = len(data)
        

        self.seqlen = seqlen

    
        resize_H = 256
        resize_W = 256
        
        for index in range(self.dict_num):
            n_frames = int(data[index]['video'].shape[0])
            self.dict_lens.append(n_frames // seqlen)
        
 
            
            video = resize_video(data[index]['video'],(resize_H,resize_W)) # S, H, W, C
            
            
            target_point = data[index]['points']
            
            target_point *= np.array([
                resize_W,
                resize_H
            ])  # N, S ,2
            
            

            
            
            occluded = data[index]['occluded']  # N, S
            

            
            self.data_to_gt.append( {
                'video': np.copy(video),
                'occluded': np.copy(occluded),
                'target_point': np.copy(target_point)
            })
            

    def __getitem__(self, index):
        # identify which sample and which starting frame it is
        subfolder_id = 0
        if self.mode == 'seqlen':
            while index >= self.dict_lens[subfolder_id]:
                index -= self.dict_lens[subfolder_id]
                subfolder_id += 1

            # start from subfolder and the frame
            
            start_frame = index * self.seqlen

            # get gt
            S = self.seqlen
            rgb_s = self.data_to_gt[subfolder_id]['video'][start_frame:start_frame+S,:,:,:] # S ,H, W, C
            occ_s = self.data_to_gt[subfolder_id]['occluded'][:,start_frame:start_frame+S] # N, S
            traj_s = self.data_to_gt[subfolder_id]['target_point'][:,start_frame:start_frame+S,:] # N, S, 2
        
        elif self.mode == 'clip':
            subfolder_id = index
            rgb_s = self.data_to_gt[subfolder_id]['video'] # S ,H, W, C
            occ_s = self.data_to_gt[subfolder_id]['occluded'] # N, S
            traj_s = self.data_to_gt[subfolder_id]['target_point'] # N, S, 2
            subfolder_id += 1
        else:
            print('unknown mode')
            
        
        


        sample = {
            'video': rgb_s, # (S ,H, W, C) 
            'occluded': occ_s, # (N, S)
            'target_points': traj_s, # (N, S, 2)
        }
        
        return sample

    def __len__(self):
        if self.mode == 'seqlen':     
            return sum(self.dict_lens)
        elif self.mode == 'clip':
            return len(self.dict_lens)


