import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
from typing import Iterable, Mapping, Tuple, Union
import mediapy as media
import torch.profiler as profiler
from tap_kubric import KubricDataset
from tqdm import tqdm
import saveload
import os
import time as Time
import argparse
import torch.distributed as dist
import torchvision.transforms.functional as TF
import random
from einops import rearrange, repeat
from core.raft import RAFT
from torch.cuda.amp import GradScaler
import torchvision.transforms as transforms
from Pointodysseydataset import PointOdysseyDataset
from vos_video import Vos_video
from core.utils.utils import bilinear_sampler, coords_grid

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
device_ids = [0,1,2,3]



def get_args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--small", default=False)
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--stride', default=8)
    parser.add_argument('--window_length', default=4)
    parser.add_argument('--transformer', default=True)
    parser.add_argument('--depth', default=6)
    parser.add_argument('--long_memory_lambda', default=0.02)
    parser.add_argument('--real_prob', default=0.5)
    
    return parser


    
def plot_tracks(rgb, points, occluded, trackgroup=None):
  """Plot tracks with matplotlib."""
  disp = []
  cmap = plt.cm.hsv

  z_list = np.arange(
      points.shape[0]) if trackgroup is None else np.array(trackgroup)
  # random permutation of the colors so nearby points in the list can get
  # different colors
  z_list = np.random.permutation(np.max(z_list) + 1)[z_list]
  colors = cmap(z_list / (np.max(z_list) + 1))
  figure_dpi = 64

  for i in range(rgb.shape[0]):
    fig = plt.figure(
        figsize=(256 / figure_dpi, 256 / figure_dpi),
        dpi=figure_dpi,
        frameon=False,
        facecolor='w')
    ax = fig.add_subplot()
    ax.axis('off')
    ax.imshow(rgb[i])

    valid = points[:, i, 0] > 0
    valid = np.logical_and(valid, points[:, i, 0] < rgb.shape[2] - 1)
    valid = np.logical_and(valid, points[:, i, 1] > 0)
    valid = np.logical_and(valid, points[:, i, 1] < rgb.shape[1] - 1)

    colalpha = np.concatenate([colors[:, :-1], 1 - occluded[:, i:i + 1]],
                              axis=1)
    plt.scatter(
        points[valid, i, 0],
        points[valid, i, 1],
        s=3,
        c=colalpha[valid],
    )

    occ2 = occluded[:, i:i + 1]

    colalpha = np.concatenate([colors[:, :-1], occ2], axis=1)
    
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(
        fig.canvas.tostring_rgb(),
        dtype='uint8').reshape(int(height), int(width), 3)
    disp.append(np.copy(img))
    plt.close(fig)

  return np.stack(disp, axis=0)

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9,0.95),weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+10,
        pct_start=0.2, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def compute_loss(trajs_g,occ_g,flow_pred_list, occ_pred_list, index, T_this, reference_frame):
    
    '''
    trajs_g: B, N, T, 2
    occ_g : B, N, T
    '''
    trajs_g = trajs_g.squeeze(0)
    occ_g = occ_g.squeeze(0)
    eps = 1e-7
    
    T, _, H, W = occ_pred_list[0].shape
    n = len(flow_pred_list)
    
    valid = occ_g[:, reference_frame] == False
    trajs_g = trajs_g[valid]
    occ_g = occ_g[valid]
    
    mask = torch.arange(trajs_g.shape[1]) >= reference_frame
    mask = mask.unsqueeze(0)
    
    sample_point = trajs_g[:, reference_frame].unsqueeze(1) # N, 1, 2
    
    grid = sample_point.unsqueeze(0).repeat(T, 1, 1, 1)/torch.tensor([W, H], device = sample_point.device)*2-1 # T, N, 1, 2
    
    flow_g = trajs_g - sample_point # N, T, 2
    
    flow_g = flow_g[:, index:index+T_this]
    
    occ_g = occ_g[:, index:index+T_this]
    
    mask = mask[:, index:index+T_this].cuda()
    
    N, _, _ = flow_g.shape
    
    flow_loss = 0
    
    occ_loss = 0
    
    uncertainty_loss = 0

    for i in range(n):
        i_weight = 0.8**(n - i - 1)
        flow = flow_pred_list[i]
        
        flow_p = F.grid_sample(flow, grid, align_corners = True, mode='bilinear')
        flow_p = rearrange(flow_p, 't c n () -> n t c')
        
        flow_div = flow_g - flow_p
        flow_div = torch.norm(flow_div,p=2,dim=2)
        uncertainty_g = flow_div > 5
        
        zero_one = torch.zeros(N,T).to(flow_div.device)
        loss = nn.SmoothL1Loss(reduction='none', beta=4)
        flow_div = loss(flow_div, zero_one)
        
        
        flow_div = flow_div * (1 - occ_g)*i_weight*mask

        flow_div = torch.sum(flow_div)
        
        flow_loss += flow_div
        
        occ_pred = occ_pred_list[i]
    
        occ_pred = F.grid_sample(occ_pred, grid, align_corners = True, mode='bilinear') # t 2 n 1
        
        occ_p = rearrange(occ_pred[:, 0], 't n () -> n t')
        
        occ_div = torch.sum(-occ_g*torch.log(occ_p+eps)-(1-occ_g)*torch.log(1-occ_p+eps)*mask)

        occ_loss += occ_div*i_weight
        
        uncertainty_p = rearrange(occ_pred[:, 1], 't n () -> n t')
        
        uncertainty_g = uncertainty_g.float()
        
        uncertainty_div = torch.sum(-uncertainty_g*torch.log(uncertainty_p+eps)-(1-uncertainty_g)*torch.log(1-uncertainty_p+eps)*mask*(1 - occ_g))

        uncertainty_loss += uncertainty_div*i_weight
    
    total_loss = flow_loss + occ_loss*10 + uncertainty_loss*10
    
    
    total_loss = total_loss/N
    
    return total_loss





    

def width_flip(occluded,target_points,video, W):
    video_w = torch.flip(video,[4])
    target_points_w = target_points.clone()
    target_points_w[:,:,:,0] = W - target_points_w[:,:,:,0]
    occluded_w = occluded.clone()
    return {
        'occluded':occluded_w,
        'target_points':target_points_w,
        'video':video_w,
    }
    
    
    
def height_flip(occluded,target_points,video,H):
    video_h = torch.flip(video,[3])
    target_points_h = target_points.clone()
    target_points_h[:,:,:,1] = H - target_points_h[:,:,:,1]
    occluded_h = occluded.clone()
    return {
    'occluded':occluded_h,
    'target_points':target_points_h,
    'video':video_h,
    }
    
def time_flip(occluded,target_points,video):
    video_t = torch.flip(video,[1])
    target_points_t = torch.flip(target_points,[1])
    occluded_t = torch.flip(occluded,[1])
    return {
    'occluded':occluded_t,
    'target_points':target_points_t,
    'video':video_t,
    }



def rotate(video):
    B, T, _, H, W = video.shape
    center = [W//2, H//2]
    per_angle = random.randint(0, 6)
    if per_angle == 0:
        angle = torch.tensor([0]).repeat(T)
    else:
        angle = torch.arange(0,per_angle*T,step = per_angle)
    pic_list = []
    for t in range(T):
        pic = video[0,t].float()
        pic = F.pad(pic, (W//2, W//2), mode='reflect')
        pic = F.pad(rearrange(pic, 'c h w -> c w h'), (H//2, H//2), mode='reflect')
        pic = rearrange(pic, 'c w h -> c h w')
        
        pic = TF.rotate(pic, angle=int(angle[t]), center = [center[0]+W//2, center[1]+H//2])
        
        pic = pic[:,H//2:H+H//2,W//2:W+W//2]
        
        pic_list.append(pic.type(torch.uint8))
        
    video_r = torch.stack(pic_list, dim=0).unsqueeze(0)
    
    return video_r
    
    
        
        
def rotate_video(occluded,target_points,video, H, W):
    center = [W//2, H//2]
    occluded = occluded.permute(0,2,1)
    target_points = target_points.permute(0,2,1,3)
    B, T, _, _, _ = video.shape
    per_angle = random.randint(0, 6)
    if per_angle == 0:
        angle = torch.tensor([0]).repeat(T)
    else:
        angle = torch.arange(0,per_angle*T,step = per_angle)
    pic_list = []
    for t in range(T):
        pic = video[0,t].float()
        pic = F.pad(pic, (W//2, W//2), mode='reflect')
        pic = F.pad(rearrange(pic, 'c h w -> c w h'), (H//2, H//2), mode='reflect')
        pic = rearrange(pic, 'c w h -> c h w')
        
        pic = TF.rotate(pic, angle=int(angle[t]), center = [center[0]+W//2, center[1]+H//2])
        
        pic = pic[:,H//2:H+H//2,W//2:W+W//2]
        
        pic_list.append(pic.type(torch.uint8))
        
    video_r = torch.stack(pic_list, dim=0).unsqueeze(0)
    

    
    target_points_r = target_points.clone()
    target_points_r[..., 0] = -(target_points[...,1] - center[1])*torch.sin(-angle/180*math.pi) + (target_points[...,0] - center[0])*torch.cos(-angle/180*math.pi) + center[0]
    target_points_r[..., 1] = (target_points[...,1] - center[1])*torch.cos(-angle/180*math.pi) + (target_points[...,0] - center[0])*torch.sin(-angle/180*math.pi) + center[1]
    
    
    occluded_r = occluded.clone()
    no_out_pic = (target_points_r[...,0] <= W) * (target_points_r[...,0] >= 0) * (target_points_r[...,1] <= H) * (target_points_r[...,1] >= 0)

    occluded_r[~no_out_pic] = True

    return {
    'occluded':occluded_r.permute(0,2,1),
    'target_points':target_points_r.permute(0,2,1,3),
    'video':video_r,
    }



        

def prob_aug(data, H, W):
    hori = torch.rand(1)
    vert = torch.rand(1)
    time = torch.rand(1)
    rotate = torch.rand(1)
    eraser = torch.rand(1)
    
    if hori < 0.5:
        data = width_flip(data['occluded'],data['target_points'],data['video'], W)
    
    if vert < 0.5:
        data = height_flip(data['occluded'],data['target_points'],data['video'], H)
            
    if time < 0.5:
        data = time_flip(data['occluded'],data['target_points'],data['video'])
        
    if rotate < 0.5:
        data = rotate_video(data['occluded'],data['target_points'],data['video'], H, W)
            
    return data

class color_normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]), requires_grad = False)
        self.std = nn.Parameter(torch.tensor([0.228, 0.224, 0.225]), requires_grad = False)
        self.prob_color_augment = 0.8
        self.prob_color_drop = 0.2
        self.prob_blur = 0.25
        self.prob_asymmetric_color_augment = 0.2
        self.photo_aug = transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.4, hue= 0.2)
        
    def forward(self, video):
        
        video = video/255.0

        asymmetric_color_aug = torch.rand(1)
        color_augment = torch.rand(1)
        color_drop = torch.rand(1)
        color_blur = torch.rand(1)
        
        _, T, _, _, _ = video.shape
        
        if color_augment < self.prob_color_augment:
            
            if asymmetric_color_aug < self.prob_asymmetric_color_augment:
                
                video_list = []
                
                for t in range(T):
                    
                    pic = self.photo_aug(video[:, t:t+1])
                    
                    video_list.append(pic)
                    
                video = torch.cat(video_list, dim = 1)
                video = torch.clip(video, 0, 1)
                
            else:
        
                video = self.photo_aug(video)
                video = torch.clip(video, 0, 1)
        
        video = video*255
        video = video.type(torch.uint8)
        
        
        return video
        

def train(height = 384,
          width = 512,
          train_epoch = 5000,
          shuffle = True,
          B = 1,
          checkpoints_dir = 'checkpoints/',
          eval_save_every_epoch = 1000,
          load_checkpoint = False,
          best_checkpoint = True,
          need_video = False,
          train_data = 'k+v',
          ):

    checkpoints_dir = 'checkpoints_window_4'
    args = get_args_parser().parse_args()
    


    args.local_rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group(
        backend='nccl', init_method=args.dist_url,
    )
    
    
    device=torch.device("cuda", args.local_rank)
    
    torch.cuda.set_device(args.local_rank)
    
    window_length = args.window_length

    model = RAFT(args = args)
    
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    optimizer,scheduler = fetch_optimizer(0.0005,0.0005, 1e-7, train_epoch, model.parameters())
    
    scaler = GradScaler(enabled=args.mixed_precision)

    if train_data == 'kubric':
    
        train_dataset = KubricDataset(mode = 'clip',seqlen = 1,dataset = 'train_with_query')
        
    elif train_data == 'pointody':
        train_dataset = PointOdysseyDataset(dset='TRAIN',
                                        use_augs=True,
                                        seq_len=100,
                                        traj_per_sample=10000,
                                        crop_size=(height, width),)
    
    elif train_data == 'k+v':
        train_dataset_k = KubricDataset(mode = 'clip',seqlen = 1,dataset = 'train_with_query')
        train_dataset_v = Vos_video(JPEGPath = '../vos/train_all_frames_zip/train_all_frames/JPEGImages', 
                                AnnPath = '../vos/train/Annotations', 
                                json_path = '../vos/train/meta.json', seqlen = 24, height = height, width = width)
        
    if train_data == 'k+v':
        
        train_sampler_k = torch.utils.data.distributed.DistributedSampler(train_dataset_k)
        
        Batch = B

        train_dataloader_k = DataLoader(
            train_dataset_k,
            batch_size=Batch,
            shuffle=False,
            num_workers=12*Batch,
            drop_last = True,
            pin_memory=True,
            sampler=train_sampler_k,
            )
        
        train_iter_k = iter(train_dataloader_k)
        
        train_sampler_v = torch.utils.data.distributed.DistributedSampler(train_dataset_v)
        
        Batch = B

        train_dataloader_v = DataLoader(
            train_dataset_v,
            batch_size=Batch,
            shuffle=False,
            num_workers=12*Batch,
            drop_last = True,
            pin_memory=True,
            sampler=train_sampler_v,
            )
        
        train_iter_v = iter(train_dataloader_v)
        
        
    else:
    
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        
        Batch = B

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=Batch,
            shuffle=False,
            num_workers=12*Batch,
            drop_last = True,
            pin_memory=True,
            sampler=train_sampler,
            )
        
        train_iter = iter(train_dataloader)

    validation_dataset = KubricDataset(mode = 'clip',seqlen = 1,dataset = 'validation_with_query')
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=B,
        shuffle=False,
        num_workers=12,
        drop_last = True,
        pin_memory=True,
        sampler=validation_sampler,
        )
    
    
    current_epoch = 0
    
    checkpoints = torch.load('checkpoints_256_256_mask_rotate/model_epoch_5000_loss_104.113.pth', map_location='cpu')
    pretrained_dict = checkpoints["model"]
    model.module.load_state_dict(pretrained_dict, strict = False)
    
    if load_checkpoint == True:
        current_epoch = saveload.load_checkpoint(checkpoints_dir, model.module, optimizer, scheduler, best_loss = best_checkpoint)

    
    color_aug = color_normalize()
    
    
    while current_epoch < train_epoch:
        if dist.get_rank()==0:
            print('TRAIN from epoch: {}'.format(current_epoch))
        
        for i in tqdm(range(eval_save_every_epoch)):
            
            optimizer.zero_grad()
            
            dist.barrier()
            
            model.train()
        
            if train_data == 'pointody':
        
                try:
                    data, gotit = next(train_iter)
                except StopIteration:
                    dist.barrier()
                    train_sampler.set_epoch(current_epoch)
                    train_iter = iter(train_dataloader)
                    data, gotit = next(train_iter)
                    
                if gotit == False:
                    scheduler.step()
                    continue
            
            elif train_data == 'kubric':
            
                try:
                    data = next(train_iter)
                except StopIteration:
                    dist.barrier()
                    train_sampler.set_epoch(current_epoch)
                    train_iter = iter(train_dataloader)
                    data = next(train_iter)
                    
            elif train_data == 'vos':
            
                try:
                    data = next(train_iter)
                except StopIteration:
                    dist.barrier()
                    train_sampler.set_epoch(current_epoch)
                    train_iter = iter(train_dataloader)
                    data = next(train_iter)
                    
            elif train_data == 'k+v':
                
                torch.random.manual_seed(current_epoch)
                poss = torch.rand(1)
                random.seed(current_epoch)

                if poss < args.real_prob:
            
                    try:
                        dist.barrier()
                        data, gotit = next(train_iter_v)

                    except StopIteration:
                        dist.barrier()
                        train_sampler_v.set_epoch(current_epoch)
                        train_iter_v = iter(train_dataloader_v)
                        data, gotit = next(train_iter_v)
                        
                else:
                    try:
                        dist.barrier()
                        data = next(train_iter_k)
                    except StopIteration:
                        dist.barrier()
                        train_sampler_k.set_epoch(current_epoch)
                        train_iter_k = iter(train_dataloader_k)
                        data = next(train_iter_k)
                
            
            if train_data == 'kubric':
                data = {
                    'occluded':data['occluded'].permute(0,2,1),
                    'target_points':data['target_points'].permute(0,2,1,3),
                    'video':data['video'].permute(0,1,4,2,3),
                }
                data_process = prob_aug(data, 256, 256)
                
                rgbs = data_process['video'].cuda().float() # B, T, C, H, W
                trajs_g = data_process['target_points'].permute(0,2,1,3).cuda().float() # B, N, T, 2
                occ_g = data_process['occluded'].permute(0,2,1).cuda().float() #  B, N, T
                
                rgbs = F.interpolate(rgbs.squeeze(0), size = (height, width), mode='bilinear').float().unsqueeze(0)
                trajs_g = trajs_g*torch.tensor([width/256, height/256]).cuda()

                
            elif train_data == 'pointody':
                data_process = prob_aug(data, height, width)
                rgbs = data_process['video'].cuda().float() # B, T, C, H, W
                trajs_g = data_process['target_points'].permute(0,2,1,3).cuda().float() # B, N, T, 2
                occ_g = data_process['occluded'].permute(0,2,1).cuda().float() #  B, N, T
                
                
            elif train_data == 'k+v':
                
                if poss < args.real_prob:
                    
                    B, T, H, W, C = data['video'].shape
                    
                    rgbs = data['video'].permute(0,1,4,2,3)
                    mask = data['mask'].squeeze(0).squeeze(-1) # 1, N, H, W, 1
                    mask_id = data['id']
                    
                    if torch.rand(1) < 0.5:
                        rgbs = torch.flip(rgbs,[3])
                        mask = torch.flip(mask,[1])
                        
                    if torch.rand(1) < 0.5:
                        rgbs = torch.flip(rgbs,[4])
                        mask = torch.flip(mask,[2])
                        
                    if torch.rand(1) < 0.5:
                        rgbs = rotate(rgbs)
                    
                    rgbs = rgbs.cuda().float() 
                    mask = mask.cuda()
                    
                else:
                    
                    data = {
                    'occluded':data['occluded'].permute(0,2,1),
                    'target_points':data['target_points'].permute(0,2,1,3),
                    'video':data['video'].permute(0,1,4,2,3),
                    }
                    
                    data_process = prob_aug(data, 256, 256)
                
                    rgbs = data_process['video'].cuda().float() # B, T, C, H, W
                    trajs_g = data_process['target_points'].permute(0,2,1,3).cuda().float() # B, N, T, 2
                    occ_g = data_process['occluded'].permute(0,2,1).cuda().float() #  B, N, T
                    
                    rgbs = F.interpolate(rgbs.squeeze(0), size = (height, width), mode='bilinear').float().unsqueeze(0)
                    trajs_g = trajs_g*torch.tensor([width/256, height/256]).cuda()
                
                
            
            if not(train_data == 'k+v' and poss < args.real_prob):
                rgbs = color_aug(rgbs)

            _, T, _, _, _ = rgbs.shape
            

            if train_data != 'k+v':
                
                index = 0
            
                random_frame = 0
                
                flow_init = None
                
                reference_frame = rgbs[:, random_frame:random_frame+1]
                
                total_loss = 0
                
                new_feature_map = None
                dict = None

                while index < T - window_length//2:
                    
                    rgbs_this = rgbs[:, index : index + window_length]
                    
                    T_this = rgbs_this.shape[1]
                    
                    if T_this < window_length:
                        add_frame_num = window_length - T_this
                        rgbs_this = torch.cat([rgbs_this, rgbs_this[:,-1:].repeat(1,add_frame_num,1,1,1)], dim = 1)

                    flow_pred_list, occ_pred_list, flow_down, new_feature_map, dict = model(rgbs_this,  reference_frame, flow_init = flow_init, win_leng = T_this, new_feature_map = new_feature_map, dict = dict, index = index)
                    
                    if index < random_frame:
                        flow_init = None
                    else:
                        flow_init = torch.cat([flow_down[-window_length//2:], flow_down[-1].repeat(window_length//2, 1, 1, 1)], dim = 0)
                    

                    loss = compute_loss(trajs_g,occ_g,flow_pred_list, occ_pred_list, index, T_this, random_frame)/10


                    scaler.scale(loss).backward()

                    
                    dist.all_reduce(loss.div_(torch.cuda.device_count()))

                    index = index + window_length//2
                    
                    total_loss = total_loss + loss.item()/2
                    
                    

                scaler.unscale_(optimizer)                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
            
            elif train_data == 'k+v':
                
                if poss < args.real_prob:
                    
                    with torch.no_grad():
                        
                        flow_list_for = []
                        occ_list_for = []
                        
                        index = 0
                        flow_init = None
                        
                        reference_frame = rgbs[:, 0:1]
                        new_feature_map = None
                        dict = None
                    
                        while index < T - window_length//2:
                            
                            rgbs_this = rgbs[:, index : index + window_length]
                
                            flow_pred_list, occ_pred_list, flow_down, new_feature_map, dict = model(rgbs_this,  reference_frame, flow_init = flow_init, win_leng = window_length, new_feature_map = new_feature_map, dict = dict, index = index)
                            
                            flow_init = torch.cat([flow_down[-window_length//2:], flow_down[-1].repeat(window_length//2, 1, 1, 1)], dim = 0)
                            
                            flow_list_for.append(flow_pred_list[-1])
                            occ_list_for.append(occ_pred_list[-1])
                            
                            index = index + window_length//2
                            
                    
                    rgbs = torch.flip(rgbs, dims = [1])
                    rgbs = color_aug(rgbs)
                                        
                    index = 0
                    
                    flow_init = None
                    new_feature_map = None
                    dict = None
                    
                    total_loss = 0
                    
                    list_index = 1
                    
                    reference_frame = rgbs[:, 0:1]

                    while index < T - window_length//2:
                        
                        rgbs_this = rgbs[:, index : index + window_length]

                        flow_pred_list_back, occ_pred_list_back, flow_down, new_feature_map, dict = model(rgbs_this,  reference_frame, flow_init = flow_init, win_leng = window_length, new_feature_map = new_feature_map, dict = dict, index = index)
                        flow_init = torch.cat([flow_down[-window_length//2:], flow_down[-1].repeat(window_length//2, 1, 1, 1)], dim = 0)
                        
                        coords = coords_grid(1, height, width, device=rgbs_this.device) # 1, 2, H, W
                        
                        sample_grid = coords + flow_list_for[-1][-1:].detach()

                        valid = (occ_list_for[-list_index][-1:,:1].detach() < 0.2)*(occ_list_for[-1][-1:,:1].detach() < 0.2) # 1, H, W

                        valid = valid *(sample_grid[0:1:,0]< width)* (sample_grid[0:1:,0] > 0) * (sample_grid[0:1:,1] < height) * (sample_grid[0:1:,1] > 0)
                        
                        valid = valid * mask.bool()        
                                               
                        valid = valid.reshape(-1)
                        
                        sample_grid  = sample_grid.permute(0,2,3,1)/torch.tensor([width-1, height-1], device = sample_grid.device)*2-1
                            
                        if torch.sum(valid) != 0:
                            loss = 0
                            n = len(flow_pred_list_back)
                            for i in range(n):
                                i_weight = 0.8**(n - i - 1)
                                flow = flow_pred_list_back[i][-1:]
                                
                                flow_sample = F.grid_sample(flow, sample_grid, align_corners = True, mode='bilinear') # 1, 2, H, W
                                
                                
                                dx = (flow_list_for[-1][-1:] - flow_list_for[-list_index][:1]).detach() + flow_sample
                                
                                dx = dx.reshape(2, -1)
                                
                                dx = dx[:,valid]
                                
                                flow_loss = torch.norm(dx,p=2,dim=0)
                                
                                loss = loss + flow_loss * i_weight
                                
                        else:
                            
                            loss = flow_pred_list_back[0][-1:] - flow_pred_list_back[0][-1:]
                            
                            loss = loss.reshape(2, -1)
                            
                            loss = torch.norm(loss,p=2,dim=0)

                        
                        loss = loss.mean()
                        loss = loss*8
    
                        scaler.scale(loss).backward()

                        dist.all_reduce(loss.div_(torch.cuda.device_count()))

                        total_loss = total_loss + loss.item()/2
                        
                        index = index + window_length//2
                        
                        list_index = list_index + 1

                    scaler.unscale_(optimizer)                
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()
                    
                    
                else:
                    
                    index = 0
                
                    random_frame = 0
                    
                    flow_init = None
                    
                    reference_frame = rgbs[:, random_frame:random_frame+1]
                    
                    total_loss = 0
                    
                    new_feature_map = None
                    dict = None

                    while index < T - window_length//2:
                        
                        rgbs_this = rgbs[:, index : index + window_length]
                        
                        T_this = rgbs_this.shape[1]
                        
                        if T_this < window_length:
                            add_frame_num = window_length - T_this
                            rgbs_this = torch.cat([rgbs_this, rgbs_this[:,-1:].repeat(1,add_frame_num,1,1,1)], dim = 1)

                        flow_pred_list, occ_pred_list, flow_down, new_feature_map, dict = model(rgbs_this,  reference_frame, flow_init = flow_init, win_leng = T_this, new_feature_map = new_feature_map, dict = dict, index = index)
                        
                        if index < random_frame:
                            flow_init = None
                        else:
                            flow_init = torch.cat([flow_down[-window_length//2:], flow_down[-1].repeat(window_length//2, 1, 1, 1)], dim = 0)
                        

                        loss = compute_loss(trajs_g,occ_g,flow_pred_list, occ_pred_list, index, T_this, random_frame)

                        scaler.scale(loss).backward()
                        
                        dist.all_reduce(loss.div_(torch.cuda.device_count()))

                        index = index + window_length//2
                        
                        total_loss = total_loss + loss.item()/2
                        
                        

                    scaler.unscale_(optimizer)                
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()
                    
            

            
            if current_epoch%50 == 0:
                if dist.get_rank() == 0:
                    print('TRIAN_LOSS_EPOCH_{}:{}'.format(current_epoch, total_loss))
                    
                    
                    if need_video == True:
                        for x in range(B):
                            disp_gt = plot_tracks(rgbs[x].permute(0,2,3,1).cpu().numpy()/255, torch.ones(T, 1, 2).permute(1,0,2).numpy(),
                                    torch.ones(T, 1).permute(1,0).numpy())
                            media.write_video('train_video/{}_{}_gt.mp4'.format(current_epoch,x), disp_gt, fps=1)
            
            current_epoch = current_epoch + 1
            
        model.eval()
        
        print('START validation:')
        

        with torch.no_grad():
            
            total_loss =0
            i = 0
            for data in tqdm(validation_dataloader):
                
                data_process = data
                
                rgbs = data_process['video'].permute(0,1,4,2,3).cuda().float() # B, T, C, H, W
                trajs_g = data_process['target_points'].cuda().float() # B, T, N, 2
                occ_g = data_process['occluded'].cuda().float() #  B, T, N
                query_points = data_process['query_points'].cuda().float() # B, N , 3
                
                rgbs = F.interpolate(rgbs.squeeze(0), size = (height, width), mode='bilinear').float().unsqueeze(0)
                trajs_g = trajs_g*torch.tensor([width/256, height/256]).cuda()
                
                
                _, T, _, _, _ = rgbs.shape
                
                if T%window_length != 0:
                    add_frame_num = window_length - T%window_length
                    rgbs = torch.cat([rgbs, rgbs[:,-1:].repeat(1,add_frame_num,1,1,1)], dim = 1)
                    
                _, T_, _, _, _ = rgbs.shape
                    
                index = 0
                
                flow_init = None
                
                reference_frame = rgbs[:, 0:1]
                
                flow_all = torch.zeros(T, 2, height, width).cuda()
                occ_all = torch.zeros(T, 2, height, width).cuda()
                new_feature_map = None
                dict = None
            
                while index < T - window_length//2:
                    
                    rgbs_this = rgbs[:, index : index + window_length]
                    
                    T_this = rgbs_this.shape[1]
                    
                    if T_this < window_length:
                        add_frame_num = window_length - T_this
                        rgbs_this = torch.cat([rgbs_this, rgbs_this[:,-1:].repeat(1,add_frame_num,1,1,1)], dim = 1)

                    flow_pred_list, occ_pred_list, flow_down, new_feature_map, dict = model(rgbs_this,  reference_frame, flow_init = flow_init, win_leng = T_this, new_feature_map = new_feature_map, dict = dict, index = index)
                    
                    flow_init = torch.cat([flow_down[-window_length//2:], flow_down[-1].repeat(window_length//2, 1, 1, 1)], dim = 0)
                    
                    flow_all[index : index + T_this] = flow_pred_list[-1]
                    occ_all[index : index + T_this] = occ_pred_list[-1]
                    
                    index = index + window_length//2
                
                
                loss = compute_loss(trajs_g,occ_g,[flow_all], [occ_all], 0, T, 0)
                
                    
                dist.all_reduce(loss.div_(torch.cuda.device_count()))
                
                total_loss += loss.item()
                
                i = i + 1
                
            total_loss = total_loss/len(validation_dataloader)
            
            if dist.get_rank()==0:
                print('CURRENT_EPOCH: {}, LOSS: {:.3f}'.format(current_epoch, total_loss))
            
        model.train()
        
        if dist.get_rank()==0:
            saveload.save_checkpoint(checkpoints_dir, model.module, optimizer, scheduler,current_epoch, total_loss)
            
        dist.barrier()
    
    

        
if __name__ == '__main__':
    train()