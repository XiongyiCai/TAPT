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
from typing import Iterable, Mapping, Tuple, Union
import pickle
from time import sleep

class KineticsDataset(torch.utils.data.Dataset):
    def __init__(self, seqlen = 8,mode = 'clip'):
        
        
        data = []
        for i in range(11):
            path = '../tapvid_kinetics/tapvid_kinetics_'+str(i).rjust(5,"0")+'.pkl'
            with open(path, 'rb') as f:
                data_temp = pickle.load(f)
                data = data + data_temp
                    
        self.mode = mode
        self.dict_lens = 24//seqlen #3

        self.dict_num = len(data)
        
        self.data_to_gt = data
        self.seqlen = seqlen


            

    def __getitem__(self, index):
        if self.mode == 'seqlen':
            subfolder_id = index//self.dict_lens

            # start from subfolder and the frame
            
            start_frame = (index-self.dict_lens*subfolder_id) * self.seqlen

            # get gt
            S = self.seqlen
            video = self.data_to_gt[subfolder_id]['video'][start_frame:start_frame+S,:,:,:] # T ,H, W, C
            occluded = self.data_to_gt[subfolder_id]['occluded'][:,start_frame:start_frame+S] # N, T
            target_points = self.data_to_gt[subfolder_id]['target_points'][:, start_frame:start_frame+S, :] # N, T,2
        
        elif self.mode == 'clip':
            subfolder_id = index
            video = self.data_to_gt[subfolder_id]['video'] # T ,H, W, C
            occluded = self.data_to_gt[subfolder_id]['occluded'] # N, T
            target_points = self.data_to_gt[subfolder_id]['target_points'] # N, T,2
        else:
            print('unknown mode')
            
        
        


        sample = {
            'video': video, # T ,H, W, C
            'occluded': occluded, # N, T
            'target_points': target_points, # N, T,2
        }

        return sample

    def __len__(self):
        if self.mode == 'seqlen':     
            return self.dict_num*self.dict_lens
        elif self.mode == 'clip':
            return self.dict_num


