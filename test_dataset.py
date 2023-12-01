import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
# import tensorflow_datasets as tfds
from typing import Iterable, Mapping, Tuple, Union
# import tensorflow as tf
import mediapy as media
import torch.profiler as profiler
from tap_kubric import KubricDataset
from tqdm import tqdm
import saveload
import os
import time as Time
from tap_davis import DAVISDataset
from tap_rgb import RGBDataset
import torchvision.transforms as transforms
from tap_kinetics import KineticsDataset
import torchvision.transforms.functional as TF
import random
import argparse
from core.raft import RAFT
from einops import rearrange, repeat
import cv2
import torch.nn.functional as F
import pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
device_ids = [0,]


def get_args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--small", default=False)
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--stride', default=8)
    parser.add_argument('--window_length', default=4)
    parser.add_argument('--transformer', default=True)
    parser.add_argument('--depth', default=6)
    parser.add_argument('--long_memory_lambda', default=0.02)
    return parser

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
def plot_tracks(rgb, points_pred, occluded_pred, points_gt, occluded_gt, trackgroup=None):
  """Plot tracks with matplotlib."""
  disp = []
  cmap = plt.cm.hsv

  z_list = np.arange(
      points_pred.shape[0]) if trackgroup is None else np.array(trackgroup)
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

    '''
    gt_point
    '''
    valid = points_gt[:, i, 0] > 0
    valid = np.logical_and(valid, points_gt[:, i, 0] < rgb.shape[2] - 1)
    valid = np.logical_and(valid, points_gt[:, i, 1] > 0)
    valid = np.logical_and(valid, points_gt[:, i, 1] < rgb.shape[1] - 1)

    colalpha = np.concatenate([colors[:, :-1], 1 - occluded_gt[:, i:i + 1]],
                              axis=1)
    plt.scatter(
        points_gt[valid, i, 0],
        points_gt[valid, i, 1],
        s=3,
        c=colalpha[valid],
    )
    
    '''
    pred_point
    '''
    
    valid = points_pred[:, i, 0] > 0
    valid = np.logical_and(valid, points_pred[:, i, 0] < rgb.shape[2] - 1)
    valid = np.logical_and(valid, points_pred[:, i, 1] > 0)
    valid = np.logical_and(valid, points_pred[:, i, 1] < rgb.shape[1] - 1)

    colalpha = np.concatenate([colors[:, :-1], 1 - occluded_gt[:, i:i + 1]],
                              axis=1)
    plt.scatter(
        points_pred[valid, i, 0],
        points_pred[valid, i, 1],
        s=30,
        c=colalpha[valid],
        marker='+', 
    )

    # occ2 = occluded_pred[:, i:i + 1]

    # colalpha = np.concatenate([colors[:, :-1], occ2], axis=1)

    # plt.scatter(
    #     points_pred[valid, i, 0],
    #     points_pred[valid, i, 1],
    #     s=20,
    #     facecolors='none',
    #     edgecolors=colalpha[valid],
        
    # )

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

# def compute_tapvid_metrics(
#     query_points: np.ndarray,
#     gt_occluded: np.ndarray,
#     gt_tracks: np.ndarray,
#     pred_occluded: np.ndarray,
#     pred_tracks: np.ndarray,
#     query_mode: str,
# ) -> Mapping[str, np.ndarray]:
#   """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
#   See the TAP-Vid paper for details on the metric computation.  All inputs are
#   given in raster coordinates.  The first three arguments should be the direct
#   outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
#   The paper metrics assume these are scaled relative to 256x256 images.
#   pred_occluded and pred_tracks are your algorithm's predictions.
#   This function takes a batch of inputs, and computes metrics separately for
#   each video.  The metrics for the full benchmark are a simple mean of the
#   metrics across the full set of videos.  These numbers are between 0 and 1,
#   but the paper multiplies them by 100 to ease reading.
#   Args:
#      query_points: The query points, an in the format [t, y, x].  Its size is
#        [b, n, 3], where b is the batch size and n is the number of queries
#      gt_occluded: A boolean array of shape [b, n, t], where t is the number
#        of frames.  True indicates that the point is occluded.
#      gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
#        in the format [x, y]
#      pred_occluded: A boolean array of predicted occlusions, in the same
#        format as gt_occluded.
#      pred_tracks: An array of track predictions from your algorithm, in the
#        same format as gt_tracks.
#      query_mode: Either 'first' or 'strided', depending on how queries are
#        sampled.  If 'first', we assume the prior knowledge that all points
#        before the query point are occluded, and these are removed from the
#        evaluation.
#   Returns:
#       A dict with the following keys:
#       occlusion_accuracy: Accuracy at predicting occlusion.
#       pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
#         predicted to be within the given pixel threshold, ignoring occlusion
#         prediction.
#       jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
#         threshold
#       average_pts_within_thresh: average across pts_within_{x}
#       average_jaccard: average across jaccard_{x}
#   """

#   metrics = {}

#   # Don't evaluate the query point.  Numpy doesn't have one_hot, so we
#   # replicate it by indexing into an identity matrix.
#   one_hot_eye = np.eye(gt_tracks.shape[2])
#   query_frame = query_points[..., 0]
#   query_frame = np.round(query_frame).astype(np.int32)
#   evaluation_points = one_hot_eye[query_frame] == 0

#   # If we're using the first point on the track as a query, don't evaluate the
#   # other points.
#   if query_mode == 'first':
#     for i in range(gt_occluded.shape[0]):
#       index = np.where(gt_occluded[i] == 0)[0][0]
#       evaluation_points[i, :index] = False
#   elif query_mode != 'strided':
#     raise ValueError('Unknown query mode ' + query_mode)

#   # Occlusion accuracy is simply how often the predicted occlusion equals the
#   # ground truth.
#   occ_acc = np.sum(
#       np.equal(pred_occluded, gt_occluded) & evaluation_points,
#       axis=(1, 2),
#   ) / np.sum(evaluation_points)
#   metrics['occlusion_accuracy'] = occ_acc

#   # Next, convert the predictions and ground truth positions into pixel
#   # coordinates.
#   visible = np.logical_not(gt_occluded)
#   pred_visible = np.logical_not(pred_occluded)
#   all_frac_within = []
#   all_jaccard = []
#   for thresh in [1, 2, 4, 8, 16]:
#     # True positives are points that are within the threshold and where both
#     # the prediction and the ground truth are listed as visible.
#     within_dist = np.sum(
#         np.square(pred_tracks - gt_tracks),
#         axis=-1,
#     ) < np.square(thresh)
#     is_correct = np.logical_and(within_dist, visible)

#     # Compute the frac_within_threshold, which is the fraction of points
#     # within the threshold among points that are visible in the ground truth,
#     # ignoring whether they're predicted to be visible.
#     count_correct = np.sum(
#         is_correct & evaluation_points,
#         axis=(1, 2),
#     )
#     count_visible_points = np.sum(
#         visible & evaluation_points, axis=(1, 2))
#     frac_correct = count_correct / count_visible_points
#     metrics['pts_within_' + str(thresh)] = frac_correct
#     all_frac_within.append(frac_correct)

#     true_positives = np.sum(
#         is_correct & pred_visible & evaluation_points, axis=(1, 2))

#     # The denominator of the jaccard metric is the true positives plus
#     # false positives plus false negatives.  However, note that true positives
#     # plus false negatives is simply the number of points in the ground truth
#     # which is easier to compute than trying to compute all three quantities.
#     # Thus we just add the number of points in the ground truth to the number
#     # of false positives.
#     #
#     # False positives are simply points that are predicted to be visible,
#     # but the ground truth is not visible or too far from the prediction.
#     gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
#     false_positives = (~visible) & pred_visible
#     false_positives = false_positives | ((~within_dist) & pred_visible)
#     false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
#     jaccard = true_positives / (gt_positives + false_positives)
#     metrics['jaccard_' + str(thresh)] = jaccard
#     all_jaccard.append(jaccard)
#   metrics['average_jaccard'] = np.mean(
#       np.stack(all_jaccard, axis=1),
#       axis=1,
#   )
#   metrics['average_pts_within_thresh'] = np.mean(
#       np.stack(all_frac_within, axis=1),
#       axis=1,
#   )
#   return metrics

def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
  """Computes TAP-Vid metrics (Jaccard, Pts.

  Within Thresh, Occ.

  Acc.)

  See the TAP-Vid paper for details on the metric computation.  All inputs are
  given in raster coordinates.  The first three arguments should be the direct
  outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
  The paper metrics assume these are scaled relative to 256x256 images.
  pred_occluded and pred_tracks are your algorithm's predictions.

  This function takes a batch of inputs, and computes metrics separately for
  each video.  The metrics for the full benchmark are a simple mean of the
  metrics across the full set of videos.  These numbers are between 0 and 1,
  but the paper multiplies them by 100 to ease reading.

  Args:
     query_points: The query points, an in the format [t, y, x].  Its size is
       [b, n, 3], where b is the batch size and n is the number of queries
     gt_occluded: A boolean array of shape [b, n, t], where t is the number of
       frames.  True indicates that the point is occluded.
     gt_tracks: The target points, of shape [b, n, t, 2].  Each point is in the
       format [x, y]
     pred_occluded: A boolean array of predicted occlusions, in the same format
       as gt_occluded.
     pred_tracks: An array of track predictions from your algorithm, in the same
       format as gt_tracks.
     query_mode: Either 'first' or 'strided', depending on how queries are
       sampled.  If 'first', we assume the prior knowledge that all points
       before the query point are occluded, and these are removed from the
       evaluation.

  Returns:
      A dict with the following keys:

      occlusion_accuracy: Accuracy at predicting occlusion.
      pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
        predicted to be within the given pixel threshold, ignoring occlusion
        prediction.
      jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
        threshold
      average_pts_within_thresh: average across pts_within_{x}
      average_jaccard: average across jaccard_{x}
  """

  metrics = {}

  eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
  if query_mode == 'first':
    # evaluate frames after the query frame
    query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
  elif query_mode == 'strided':
    # evaluate all frames except the query frame
    query_frame_to_eval_frames = 1 - eye
  else:
    raise ValueError('Unknown query mode ' + query_mode)

  query_frame = query_points[..., 0]
  query_frame = np.round(query_frame).astype(np.int32)
  evaluation_points = query_frame_to_eval_frames[query_frame] > 0

  # Occlusion accuracy is simply how often the predicted occlusion equals the
  # ground truth.
  occ_acc = np.sum(
      np.equal(pred_occluded, gt_occluded) & evaluation_points,
      axis=(1, 2),
  ) / np.sum(evaluation_points)
  metrics['occlusion_accuracy'] = occ_acc

  # Next, convert the predictions and ground truth positions into pixel
  # coordinates.
  visible = np.logical_not(gt_occluded)
  pred_visible = np.logical_not(pred_occluded)
  all_frac_within = []
  all_jaccard = []
  for thresh in [1, 2, 4, 8, 16]:
    # True positives are points that are within the threshold and where both
    # the prediction and the ground truth are listed as visible.
    within_dist = np.sum(
        np.square(pred_tracks - gt_tracks),
        axis=-1,
    ) < np.square(thresh)
    is_correct = np.logical_and(within_dist, visible)

    # Compute the frac_within_threshold, which is the fraction of points
    # within the threshold among points that are visible in the ground truth,
    # ignoring whether they're predicted to be visible.
    count_correct = np.sum(
        is_correct & evaluation_points,
        axis=(1, 2),
    )
    count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
    frac_correct = count_correct / count_visible_points
    metrics['pts_within_' + str(thresh)] = frac_correct
    all_frac_within.append(frac_correct)

    true_positives = np.sum(
        is_correct & pred_visible & evaluation_points, axis=(1, 2)
    )

    # The denominator of the jaccard metric is the true positives plus
    # false positives plus false negatives.  However, note that true positives
    # plus false negatives is simply the number of points in the ground truth
    # which is easier to compute than trying to compute all three quantities.
    # Thus we just add the number of points in the ground truth to the number
    # of false positives.
    #
    # False positives are simply points that are predicted to be visible,
    # but the ground truth is not visible or too far from the prediction.
    gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
    false_positives = (~visible) & pred_visible
    false_positives = false_positives | ((~within_dist) & pred_visible)
    false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
    jaccard = true_positives / (gt_positives + false_positives)
    metrics['jaccard_' + str(thresh)] = jaccard
    all_jaccard.append(jaccard)
  metrics['average_jaccard'] = np.mean(
      np.stack(all_jaccard, axis=1),
      axis=1,
  )
  metrics['average_pts_within_thresh'] = np.mean(
      np.stack(all_frac_within, axis=1),
      axis=1,
  )
  return metrics

def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.
    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.
    Args:
        target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
        target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
        frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
    Returns:
        A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
            each point is [t, y, x] scaled to the range [-1, 1]
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
            each point is [x, y] scaled to the range [-1, 1]
    """
    video_list = []
    query_points_list = []
    target_points_list = []
    occluded_list = []
    
    target_occluded = target_occluded.squeeze(0)
    target_points = target_points.squeeze(0)
    frames = frames.squeeze(0)
    
    valid = torch.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = torch.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(torch.tensor([index, y, x]))  # [t, y, x]
    query_points = torch.stack(query_points, axis=0)

    return {
      'video': frames.unsqueeze(0),
      'query_points': query_points.unsqueeze(0),
      'target_points': target_points.unsqueeze(0),
      'occluded': target_occluded.unsqueeze(0),
    }
    
    
def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
    ) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.

    Given a set of frames and tracks with no query points, sample queries
    strided every query_stride frames, ignoring points that are not visible
    at the selected frames.

    Args:
        target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
        target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
        frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
        query_stride: When sampling query points, search for un-occluded points
        every query_stride frames and convert each one into a query.

    Returns:
        A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3].  The video
            has floats scaled to the range [-1, 1].
        query_points: Query points of shape [1, n_queries, 3] where
            each point is [t, y, x] scaled to the range [-1, 1].
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
            each point is [x, y] scaled to the range [-1, 1].
        trackgroup: Index of the original track that each query point was
            sampled from.  This is useful for visualization.
    """
    tracks = []
    occs = []
    queries = []
    
    target_occluded = target_occluded.squeeze(0)
    target_points = target_points.squeeze(0)
    frames = frames.squeeze(0)

    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = torch.stack(
            [
                i * torch.ones(target_occluded.shape[0:1]), target_points[:, i, 1],
                target_points[:, i, 0]
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
    return {
      'video':
          frames.unsqueeze(0),
      'query_points':
          torch.cat(queries, axis=0).unsqueeze(0),
      'target_points':
          torch.cat(tracks, axis=0).unsqueeze(0),
      'occluded':
          torch.cat(occs, axis=0).unsqueeze(0),
    }

def rotate_video(occluded,target_points,video,query_points, H, W):
    center = [W/2, H/2]
    B, T, _, _, _ = video.shape
    video = video.permute(0,1,4,2,3)
    B, N, _ = query_points.shape
    per_angle = random.randint(0,8)
    if per_angle == 0:
        angle = torch.tensor([0]).repeat(T)
    else:
        angle = torch.arange(0,per_angle*T,step = per_angle)
    # angle = torch.tensor([0]).repeat(T)
    pic_list = []
    for t in range(T):
        pic = video[0,t].float()
        pic = F.pad(pic, (W//2, W//2), mode='reflect')
        pic = F.pad(rearrange(pic, 'c h w -> c w h'), (H//2, H//2), mode='reflect')
        pic = rearrange(pic, 'c w h -> c h w')
        pic = TF.rotate(pic, angle=int(angle[t]), center = [center[0]+W//2, center[1]+H//2])
        
        pic = pic[:,H//2:H+H//2,W//2:W+W//2]
        
        # pic = video[0, t]
        # pic = repeat(pic, 'c h w -> c (a h) (b w)', a = 3, b = 3)
        # pic = pic[:,H//2:2*H+H//2, W//2:2*W+W//2]
        
        # pic = TF.rotate(pic, angle=int(angle[t]), center = [center[0]+W//2, center[1]+H//2])
        # pic = pic[:, H//2:H+H//2, W//2:W+W//2]
        
        pic_list.append(pic.int())
        
    video_r = torch.stack(pic_list, dim=0).unsqueeze(0)
    

    
    target_points_r = target_points.clone()
    target_points_r[..., 0] = -(target_points[...,1] - center[1])*torch.sin(-angle/180*math.pi) + (target_points[...,0] - center[0])*torch.cos(-angle/180*math.pi) + center[1]
    target_points_r[..., 1] = (target_points[...,1] - center[1])*torch.cos(-angle/180*math.pi) + (target_points[...,0] - center[0])*torch.sin(-angle/180*math.pi) + center[0]
    
    # query_points_r = query_points.clone()
    # query_points_r[..., 2] = -(query_points[...,1] - center[1])*torch.sin(-angle[query_points[0,:,0].long()]/180*math.pi) + (query_points[...,2] - center[0])*torch.cos(-angle[query_points[0,:,0].long()]/180*math.pi) + center[1]
    # query_points_r[..., 1] = (query_points[...,1] - center[1])*torch.cos(-angle[query_points[0,:,0].long()]/180*math.pi) + (query_points[...,2] - center[0])*torch.sin(-angle[query_points[0,:,0].long()]/180*math.pi) + center[0]
    query_points_r = target_points_r[:,torch.arange(N),query_points[0,:,0].long()] # B, N, 2
    query_points_r = query_points_r[:,:,[1,0]]
    query_points_r = torch.cat([query_points[:,:,0:1], query_points_r], dim = 2)
    
    
    occluded_r = occluded.clone()
    no_out_pic = (target_points_r[...,0] <= W) * (target_points_r[...,0] >= 0) * (target_points_r[...,1] <= H) * (target_points_r[...,1] >= 0)

    occluded_r[~no_out_pic] = True
    
    valid = occluded_r[:,torch.arange(N),query_points_r[0,:,0].long()] == False

    return {
    'occluded':occluded_r[valid].unsqueeze(0),
    'target_points':target_points_r[valid].unsqueeze(0),
    'video':video_r.permute(0,1,3,4,2),
    'query_points' : query_points_r[valid].unsqueeze(0),
    }
    
def better_query(occluded,target_points,video,query_points):
    
    _, N, _ = query_points.shape
      
    query_points_better = target_points[:,torch.arange(N),query_points[0,:,0].long()]
    
    query_points_better = query_points_better[:,:,[1,0]]
    query_points_better = torch.cat([query_points[:,:,0:1], query_points_better], dim = 2)

    return {
    'occluded':occluded,
    'target_points':target_points,
    'video':video,
    'query_points' : query_points_better,
    }

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9,0.999),weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+10,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def PositionEmbeddingSine1D(L, C):
    pe = torch.zeros((L,C))
    position = torch.arange(0, L).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, C, 2) *
                            -(math.log(10000.0) / C))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

def PositionEmbeddingSine2D(dim,H,W):
    C = dim//2
    
    pos_x = torch.zeros((W,C))
    position = torch.arange(0, W).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, C, 2) *
                            -(math.log(10000.0) / C))
    pos_x[:, 0::2] = torch.sin(position * div_term)
    pos_x[:, 1::2] = torch.cos(position * div_term) # W, C 
    pos_x = pos_x.permute(1,0) # C, W
    pos_x = pos_x.unsqueeze(0).unsqueeze(2).repeat(1,1,H,1) # 1, C, H, W
    
    
    pos_y = torch.zeros((H,C))
    position = torch.arange(0, H).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, C, 2) *
                            -(math.log(10000.0) / C))
    pos_y[:, 0::2] = torch.sin(position * div_term)
    pos_y[:, 1::2] = torch.cos(position * div_term)
    pos_y = pos_y.permute(1,0) # C, H
    pos_y = pos_y.unsqueeze(0).unsqueeze(3).repeat(1,1,1,W) # 1, C, H,W
    pos = torch.cat((pos_y, pos_x), dim=1) # 1, 2C, H, W
    
    return pos

def make_model(dim, hidden_dim, head, depth, dropout, time, 
               factor, height, width, align_corner, time_encode, position_encode, backbone_name, patch_pos, use_time_encode, use_gate):
    model = Trackingformer(dim, hidden_dim, head, depth, dropout, time, 
                           factor, height, width, align_corner, time_encode, position_encode, backbone_name, patch_pos, use_time_encode, use_gate)
    for p in model.parameters():
        if not isinstance(p, nn.parameter.Parameter):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model

def compute_loss(trajs_g,occ_g,trajs_p,occ_p):
    eps = 1e-8
    B,S,N,_ = trajs_g.shape
    trajs_div = trajs_g - trajs_p
    trajs_div = torch.norm(trajs_div,p=2,dim=3)
    
    zero_one = torch.zeros(B,S,N).to(trajs_div.device)
    loss = nn.SmoothL1Loss(reduction='none', beta=4)
    trajs_div = loss(trajs_div, zero_one)
    
    
    trajs_div = trajs_div * (1 - occ_g)

    trajs_div = torch.sum(trajs_div)
    
    occ_div = torch.sum(-occ_g*torch.log(occ_p+eps)-(1-occ_g)*torch.log(1-occ_p+eps))
    
    total_loss = trajs_div + occ_div*10
    
    total_loss = total_loss/(N*B)

    return total_loss


def eval_problem(trajs_g,occ_g,trajs_p,occ_p,query_points):
    # B, N, T    
    
    metric = {}
    B, N, T = occ_g.shape
    True_positive_occ = np.sum(occ_g&occ_p)/(np.sum(occ_g)+1e-8)
    True_negative_occ = np.sum(~occ_g&~occ_p)/(np.sum(~occ_g)+1e-8)
    False_positive_occ = np.sum(occ_g&~occ_p)/(np.sum(occ_g)+1e-8)
    False_negative_occ = np.sum(~occ_g&occ_p)/(np.sum(~occ_g)+1e-8)
    
    
    dist = np.linalg.norm((trajs_p-trajs_g), ord=2, axis=3, keepdims = False)
    if np.sum(occ_g&occ_p) != 0:
        True_positive_dist = np.sum(np.multiply(dist,(occ_g&occ_p)))/np.sum(occ_g&occ_p)
    else:
        True_positive_dist = 0
    
    if np.sum(~occ_g&~occ_p) != 0:
        True_negative_dist = np.sum(np.multiply(dist,(~occ_g&~occ_p)))/np.sum(~occ_g&~occ_p)
    else:
        True_negative_dist = 0
        
    if np.sum(occ_g&~occ_p) != 0:
        False_positive_dist = np.sum(np.multiply(dist,(occ_g&~occ_p)))/np.sum(occ_g&~occ_p)
    else:
        False_positive_dist = 0
    
    if np.sum(~occ_g&occ_p) != 0:
        False_negative_dist = np.sum(np.multiply(dist,(~occ_g&occ_p)))/np.sum(~occ_g&occ_p)
    else:
        False_negative_dist = 0
        
    query_frame = query_points[:,:,[0]] # B, N, 1
    T_index = np.arange(T).reshape(1,1,T).repeat(B, axis = 0).repeat(N, axis = 1) # B, N, T

    
    time_diff = np.abs(T_index - query_frame)
    
    number_list = []
    True_positive_dist_list = []
    True_negative_dist_list = []
    False_positive_dist_list = []
    False_negative_dist_list = []
    
    True_positive_occ_list = []
    True_negative_occ_list = []
    False_positive_occ_list = []
    False_negative_occ_list = []
    
    for i in range(250):
        eval_points = (time_diff==i)
        number_list.append(np.sum(eval_points))
        
        True_positive_dist_list.append(np.sum(np.multiply(dist,(occ_g&occ_p&eval_points)))/(np.sum(occ_g&occ_p&eval_points)+1e-8))
        True_negative_dist_list.append(np.sum(np.multiply(dist,(~occ_g&~occ_p&eval_points)))/(np.sum(~occ_g&~occ_p&eval_points)+1e-8))
        False_positive_dist_list.append(np.sum(np.multiply(dist,(occ_g&~occ_p&eval_points)))/(np.sum(occ_g&~occ_p&eval_points)+1e-8))
        False_negative_dist_list.append(np.sum(np.multiply(dist,(~occ_g&occ_p&eval_points)))/(np.sum(~occ_g&occ_p&eval_points)+1e-8))
        
        True_positive_occ_list.append(np.sum((occ_g&occ_p&eval_points))/(np.sum(occ_g&eval_points)+1e-8))
        True_negative_occ_list.append(np.sum((~occ_g&~occ_p&eval_points))/(np.sum(~occ_g&eval_points)+1e-8))
        False_positive_occ_list.append(np.sum((occ_g&~occ_p&eval_points))/(np.sum(occ_g&eval_points)+1e-8))
        False_negative_occ_list.append(np.sum((~occ_g&occ_p&eval_points))/(np.sum(~occ_g&eval_points)+1e-8))
        
        
        
    
    
    metric['True_positive_occ'] = True_positive_occ
    metric['True_negative_occ'] = True_negative_occ
    metric['False_positive_occ'] = False_positive_occ
    metric['False_negative_occ'] = False_negative_occ
    metric['True_positive_dist'] = True_positive_dist
    metric['True_negative_dist'] = True_negative_dist
    metric['False_positive_dist'] = False_positive_dist
    metric['False_negative_dist'] = False_negative_dist
    
    metric['True_positive_dist_list'] = np.array(True_positive_dist_list)
    metric['True_negative_dist_list'] = np.array(True_negative_dist_list)
    metric['False_positive_dist_list'] = np.array(False_positive_dist_list)
    metric['False_negative_dist_list'] = np.array(False_negative_dist_list)
    
    metric['True_positive_occ_list'] = np.array(True_positive_occ_list)
    metric['True_negative_occ_list'] = np.array(True_negative_occ_list)
    metric['False_positive_occ_list'] = np.array(False_positive_occ_list)
    metric['False_negative_occ_list'] = np.array(False_negative_occ_list)
    
    return metric

def test( height = 384,
          width = 512,
          shuffle = False,
          B = len(device_ids),
          load_checkpoint = True,
          best_checkpoint = True,
          need_video = False,
          datasets = 'davis',
          test_mode = 'first',
          time = 1,
          ):

    
    assert(datasets == 'davis' or datasets == 'rgb' or datasets == 'kubric' or datasets == 'kinetics')
    
    args = get_args_parser().parse_args()
    model = RAFT(args = args)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()
    
    window_length = args.window_length
    
    
    
    if datasets == 'davis':
        test_dataset = DAVISDataset(mode = 'clip',seqlen = time)
    elif datasets == 'rgb':
        test_dataset = RGBDataset(mode = 'clip',seqlen = time)
    elif datasets == 'kubric':
        test_dataset = KubricDataset(mode = 'clip',seqlen = time,dataset = 'validation_with_query')
    elif datasets == 'kinetics':
        test_dataset = KineticsDataset(mode = 'clip',seqlen = time)
        



    TEST_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12*len(device_ids),
        drop_last = True,
        )
    
    

    
    if load_checkpoint == True:

        checkpoints = torch.load('checkpoints_window_4/model_epoch_5000_loss_112.271.pth')
        pretrained_dict = checkpoints["model"]
        model.module.load_state_dict(pretrained_dict, strict = False)


    
    # current_epoch = 0
    
    # while current_epoch < train_epoch:
        
    #     print('TRAIN from epoch: {}'.format(current_epoch))
    #     # print(optimizer.param_groups[0]['lr'])
        
    #     for i in tqdm(range(eval_save_every_epoch)):
        
    #         try:
    #             data = next(train_iter)
    #         except StopIteration:
    #             train_iter = iter(TEST_dataloader)
    #             data = next(train_iter)
                
            
    #         if data_augment == True:
    #             '''
    #             change B = 1 to B = B(max = 4)
    #             '''
    #             assert(data['target_points'].shape[0] == 1)
    #             data_process = data_AUG(data['occluded'],data['target_points'],data['video'], B = B, H = height, W = width)
    #             data_process = sample_queries_first(data_process['occluded'],data_process['target_points'],data_process['video'],max_point = number_point_limit, B = B)
    #             if torch.max(data_process['target_points']) > 1000 or  torch.min(data_process['target_points']) < -1000:
    #                 scheduler.step()
    #                 current_epoch = current_epoch + 1
    #                 continue
                
    #         else:
    #             data_process = sample_queries_first(data['occluded'],data['target_points'],data['video'],max_point = number_point_limit, B = B)
    #             if torch.max(data_process['target_points']) > 1000 or  torch.min(data_process['target_points']) < -1000:
    #                 scheduler.step()
    #                 current_epoch = current_epoch + 1
    #                 continue
                
            
            
    #         rgbs = data_process['video'].permute(0,1,4,2,3).cuda().float() # B, T, C, H, W
    #         trajs_g = data_process['target_points'].permute(0,2,1,3).cuda().float() # B, T, N, 2
    #         occ_g = data_process['occluded'].permute(0,2,1).cuda().float() #  B, T, N
    #         query_points = data_process['query_points'].cuda().float() # B, N , 3
            
    #         traj_pred, vis_pred = model(rgbs, query_points)
            
    #         traj_pred = traj_pred.permute(0,2,1,3)
    #         vis_pred = vis_pred.permute(0,2,1)
            
    #         loss = compute_loss(trajs_g,occ_g,traj_pred,vis_pred)
            
    #         loss.backward()
            
    #         optimizer.step()
            
    #         optimizer.zero_grad()
                
    #         scheduler.step()
            
    #         current_epoch = current_epoch + 1
              
    #     # torch.cuda.empty_cache()
            
    #     model.eval()
        
    #     print('START validation:')
        
    #     with torch.no_grad():
            
    #         total_loss =0
    #         for data in tqdm(validation_dataloader):
                
    #             data_process = sample_queries_first(data['occluded'],data['target_points'],data['video'],max_point = number_point_limit, B = B)
                
    #             rgbs = data_process['video'].permute(0,1,4,2,3).cuda().float() # B, T, C, H, W
    #             trajs_g = data_process['target_points'].permute(0,2,1,3).cuda().float() # B, T, N, 2
    #             occ_g = data_process['occluded'].permute(0,2,1).cuda().float() #  B, T, N
    #             query_points = data_process['query_points'].cuda().float() # B, N , 3
    #             traj_pred, vis_pred = model(rgbs, query_points)
                
    #             traj_pred = traj_pred.permute(0,2,1,3)
    #             vis_pred = vis_pred.permute(0,2,1)
                
                
    #             loss = compute_loss(trajs_g,occ_g,traj_pred,vis_pred)
    
    #             total_loss += loss.item()
                
                
    #         total_loss = total_loss/len(validation_dataloader)
    #         writer.add_scalar('Loss',total_loss,current_epoch)
    #         print('CURRENT_EPOCH: {}, LOSS: {:.3f}'.format(current_epoch, total_loss))
                
    #     model.train()
        
    #     saveload.save_checkpoint(checkpoints_dir, model.module, optimizer, scheduler,current_epoch, total_loss)
    
    torch.cuda.empty_cache()
    model.eval()
    print('START EVALUATION ON {}'.format(datasets))
    with torch.no_grad():
        i= 0
        occlusion_accuracy_list = []
        pts_within_1_list = []
        pts_within_2_list = []
        pts_within_4_list = []
        pts_within_8_list = []
        pts_within_16_list = []
        jaccard_1_list = []
        jaccard_2_list = []
        jaccard_4_list = []
        jaccard_8_list = []
        jaccard_16_list = []
        average_jaccard_list = []
        average_pts_within_thresh_list = []
        True_positive_occ_list = []
        True_negative_occ_list = []
        False_positive_occ_list = []
        False_negative_occ_list = []
        
        True_positive_dist_list = []
        True_negative_dist_list = []
        False_positive_dist_list = []
        False_negative_dist_list = []
        
        True_positive_dist_time_list = []
        True_negative_dist_time_list = []
        False_positive_dist_time_list = []
        False_negative_dist_time_list = []

        True_positive_occ_time_list = []
        True_negative_occ_time_list = []
        False_positive_occ_time_list = []
        False_negative_occ_time_list = []
        
        total_loss = 0
        i = 0
        data_list = []
        for data in tqdm(TEST_dataloader):
            
            # if i!= 5:
            #     i = i+1
            #     continue
            
            # i = i + 1
            
            if datasets == 'kubric':
                data_process = better_query(data['occluded'],data['target_points'],data['video'],data['query_points'])
                # data_process = data
            else:
                if test_mode == 'first':
                    data_process = sample_queries_first(data['occluded'],data['target_points'],data['video'])
                elif test_mode == 'strided':
                    data_process = sample_queries_strided(data['occluded'],data['target_points'],data['video'])
            
            rgbs = data_process['video'].permute(0,1,4,2,3).cuda().float() # B, T, C, H, W
            trajs_g = data_process['target_points'].cuda().float() # B, N, T, 2
            occ_g = data_process['occluded'].cuda() #  B, N, T
            query_points = data_process['query_points'].cuda().float() # B, N , 3
            
            
            # print(rgbs.shape)
            # print(query_points.shape)
            # rgbs = F.interpolate(rgbs.squeeze(0), size = (512, 512), mode='bilinear', align_corners=False).float().unsqueeze(0)
            
            rgbs = F.interpolate(rgbs.squeeze(0), size = (height, width), mode='bilinear').float().unsqueeze(0)
            trajs_g = trajs_g*torch.tensor([width/256, height/256]).cuda()
            
            query_points[:, :, 1:] = query_points[:, :, 1:]*torch.tensor([height/256, width/256]).cuda()
            
            _, T, _, _, _ = rgbs.shape
            
            # _, sort_index = torch.sort(query_points[0,:,0])
            
            # trajs_g = trajs_g[0, sort_index]
            
            # occ_g = occ_g[0, sort_index]
            
            # query_points = query_points[0, sort_index]
            # query_points[:,1:] = query_points[:,1:]
            trajs_g = trajs_g[0]
            occ_g = occ_g[0]
            query_points  = query_points[0]
            time_index = query_points[:,0]
            
            # traj_list = []
            # occ_list = []
            traj_pred_all = torch.zeros_like(trajs_g)
            occ_pred_all = torch.zeros_like(occ_g)
            
            if test_mode == 'first':
            
                for t in range(T):
                    
                    valid_frame = time_index == t
                    
                    if torch.sum(valid_frame) == 0:
                        continue
                    
                    trajs_g_t = trajs_g[valid_frame] # N, T, 2
                    occ_g_t = occ_g[valid_frame] # N, T
                    query_points_t = query_points[valid_frame]
                    query_points_t = query_points_t[:, [2,1]] # N, 2
                    
                    # flow_pred_list, occ_pred_list = model(rgbs,  t)
                    
                    # flow_pred = flow_pred_list[-1] # T, 2, H, W
                    # occ_pred = occ_pred_list[-1]
                    
                    # grid = query_points_t.unsqueeze(1).unsqueeze(0).repeat(T, 1, 1, 1)/torch.tensor([width, height], device = query_points_t.device)*2-1 # T, N, 1, 2
                    
                    # flow_p = F.grid_sample(flow_pred, grid, align_corners = False, mode='bilinear')
                    # flow_p = rearrange(flow_p, 't c n () -> n t c')
                    # traj_p = flow_p + query_points_t.unsqueeze(1)
                    
                    # occ_p = F.grid_sample(occ_pred, grid, align_corners = False, mode='bilinear')
        
                    # occ_p = rearrange(occ_p, 't () n () -> n t')
                    # occ_p = occ_p > 0.3
                    
                    # traj_list.append(traj_p)
                    # occ_list.append(occ_p)
                    rgbs_ = rgbs[:, t:]
                    
                    _, T_, _, _, _ = rgbs_.shape
                    
                    index = 0
                    
                    flow_init = None
                    
                    reference_frame = rgbs[:, t:t+1]
                    
                    flow_all = torch.zeros(T_, 2, height, width).cuda()
                    occ_all = torch.zeros(T_, 1, height, width).cuda()
                    unc_all = torch.zeros(T_, 1, height, width).cuda()
                    new_feature_map = None
                    dict = None
                    
                    while index < T_ - window_length//2:
                        
                        rgbs_this = rgbs_[:, index : index + window_length]
                        
                        T_this = rgbs_this.shape[1]
                        
                        if T_this < window_length:
                            add_frame_num = window_length - T_this
                            rgbs_this = torch.cat([rgbs_this, rgbs_this[:,-1:].repeat(1,add_frame_num,1,1,1)], dim = 1)

                        flow_pred_list, occ_pred_list, flow_down, new_feature_map, dict = model(rgbs_this,  reference_frame, flow_init = flow_init, win_leng = T_this, new_feature_map = new_feature_map, dict = dict, index = index)
                        
                        # if index != 0:
                        #     print(torch.sum(new_feature_map - new_feature_map_1))
                        # new_feature_map_1 = new_feature_map
                        
                        flow_init = torch.cat([flow_down[-window_length//2:], flow_down[-1].repeat(window_length//2, 1, 1, 1)], dim = 0)
                        
                        flow_all[index : index + T_this] = flow_pred_list[-1]
                        occ_all[index : index + T_this] = occ_pred_list[-1][:, :1]
                        unc_all[index : index + T_this] = occ_pred_list[-1][:, 1:]
                        
                        index = index + window_length//2
                        
                    flow_pred = flow_all # T, 2, H, W
                    occ_pred = occ_all
                    unc_pred = unc_all
                    
                    grid = query_points_t.unsqueeze(1).unsqueeze(0).repeat(T_, 1, 1, 1)/torch.tensor([width, height], device = query_points_t.device)*2-1 # T, N, 1, 2
                    
                    flow_p = F.grid_sample(flow_pred, grid, align_corners = False, mode='bilinear')
                    flow_p = rearrange(flow_p, 't c n () -> n t c')
                    traj_p = flow_p + query_points_t.unsqueeze(1)
                    
                    occ_p = F.grid_sample(occ_pred, grid, align_corners = False, mode='bilinear')
                    unc_p = F.grid_sample(unc_pred, grid, align_corners = False, mode='bilinear')
        
                    occ_p = rearrange(occ_p, 't () n () -> n t')
                    unc_p = rearrange(unc_p, 't () n () -> n t')
                    occ_p = occ_p > 0.3
                    
                    traj_p = traj_p*torch.tensor([256/width, 256/height]).cuda()
                    
                    # traj_list.append(traj_p)
                    # occ_list.append(occ_p)
                    traj_pred_all[valid_frame, t:] = traj_p
                    occ_pred_all[valid_frame, t:] = occ_p
                        
            elif test_mode == 'strided':
                
                for t in range(T):
                    
                    valid_frame = time_index == t
                    
                    if torch.sum(valid_frame) == 0:
                        continue
                    
                    trajs_g_t = trajs_g[valid_frame] # N, T, 2
                    occ_g_t = occ_g[valid_frame] # N, T
                    query_points_t = query_points[valid_frame]
                    query_points_t = query_points_t[:, [2,1]] # N, 2
                    
                    flow_all = torch.zeros(T, 2, height, width).cuda()
                    occ_all = torch.zeros(T, 1, height, width).cuda()
                    
                    rgbs_f = rgbs[:, t:]
                    rgbs_b = torch.flip(rgbs[:, :t+1], dims = [1])
                    
                    reference_frame = rgbs[:, t:t+1]
                    
                    _, T_, _, _, _ = rgbs_f.shape
                    
                    index = 0
                    
                    flow_init = None
                    
                    new_feature_map = None
                    dict = None
                    
                    while index < T_ - window_length//2:
                        
                        rgbs_this = rgbs_f[:, index : index + window_length]
                        
                        T_this = rgbs_this.shape[1]
                        
                        if T_this < window_length:
                            add_frame_num = window_length - T_this
                            rgbs_this = torch.cat([rgbs_this, rgbs_this[:,-1:].repeat(1,add_frame_num,1,1,1)], dim = 1)

                        flow_pred_list, occ_pred_list, flow_down, new_feature_map, dict = model(rgbs_this,  reference_frame, flow_init = flow_init, win_leng = T_this, new_feature_map = new_feature_map, dict = dict, index = index)
                        
                        # if index != 0:
                        #     print(torch.sum(new_feature_map - new_feature_map_1))
                        # new_feature_map_1 = new_feature_map
                        
                        flow_init = torch.cat([flow_down[-window_length//2:], flow_down[-1].repeat(window_length//2, 1, 1, 1)], dim = 0)
                        
                        flow_all[t + index : t + index + T_this] = flow_pred_list[-1]
                        occ_all[t + index : t + index + T_this] = occ_pred_list[-1][:, :1]
                        
                        index = index + window_length//2
                        
                    _, T_, _, _, _ = rgbs_b.shape
                    
                    index = 0
                    
                    flow_init = None
                    
                    new_feature_map = None
                    dict = None
                    
                    while index < T_ - window_length//2:
                        
                        rgbs_this = rgbs_b[:, index : index + window_length]
                        
                        T_this = rgbs_this.shape[1]
                        
                        if T_this < window_length:
                            add_frame_num = window_length - T_this
                            rgbs_this = torch.cat([rgbs_this, rgbs_this[:,-1:].repeat(1,add_frame_num,1,1,1)], dim = 1)

                        flow_pred_list, occ_pred_list, flow_down, new_feature_map, dict = model(rgbs_this,  reference_frame, flow_init = flow_init, win_leng = T_this, new_feature_map = new_feature_map, dict = dict, index = index)
                        
                        # if index != 0:
                        #     print(torch.sum(new_feature_map - new_feature_map_1))
                        # new_feature_map_1 = new_feature_map
                        
                        flow_init = torch.cat([flow_down[-window_length//2:], flow_down[-1].repeat(window_length//2, 1, 1, 1)], dim = 0)
                        
                        flow_all[t - index - T_this + 1 : t - index + 1] = torch.flip(flow_pred_list[-1], dims = [0])
                        occ_all[t - index - T_this + 1 : t - index + 1] = torch.flip(occ_pred_list[-1][:, :1], dims = [0])
                        
                        index = index + window_length//2
                    
                    
                    flow_pred = flow_all # T, 2, H, W
                    occ_pred = occ_all
                    
                    grid = query_points_t.unsqueeze(1).unsqueeze(0).repeat(T, 1, 1, 1)/torch.tensor([width, height], device = query_points_t.device)*2-1 # T, N, 1, 2
                    
                    flow_p = F.grid_sample(flow_pred, grid, align_corners = False, mode='bilinear')
                    flow_p = rearrange(flow_p, 't c n () -> n t c')
                    traj_p = flow_p + query_points_t.unsqueeze(1)
                    
                    occ_p = F.grid_sample(occ_pred, grid, align_corners = False, mode='bilinear')
        
                    occ_p = rearrange(occ_p, 't () n () -> n t')
                    occ_p = occ_p > 0.3
                    
                    traj_p = traj_p*torch.tensor([256/width, 256/height]).cuda()
                    
                    # traj_list.append(traj_p)
                    # occ_list.append(occ_p)
                    traj_pred_all[valid_frame, ] = traj_p
                    occ_pred_all[valid_frame, ] = occ_p
                        
                
            traj_pred = traj_pred_all
            # traj_pred = traj_pred/2
            occ_pred = occ_pred_all
            
            trajs_g = trajs_g*torch.tensor([256/width, 256/height]).cuda()
            
            # for x in range(25):
            #     print(x)
            #     Time.sleep(10)
            #     print(traj_pred[x])
                 
            # traj_pred = traj_pred.permute(0,2,1,3)
            # occ_pred = occ_pred.permute(0,2,1)
            
            metrics = compute_tapvid_metrics(
                                query_points= query_points.unsqueeze(0).cpu().detach().numpy(),
                                gt_occluded = occ_g.unsqueeze(0).cpu().detach().numpy(),
                                gt_tracks = trajs_g.unsqueeze(0).cpu().detach().numpy(),
                                pred_occluded = occ_pred.unsqueeze(0).cpu().detach().numpy(),
                                pred_tracks = traj_pred.unsqueeze(0).cpu().detach().numpy(),
                                query_mode = test_mode)
            
            
            test_metric = eval_problem(
                                trajs_g = trajs_g.unsqueeze(0).cpu().detach().numpy(),
                                occ_g = occ_g.unsqueeze(0).cpu().detach().numpy(),
                                trajs_p = traj_pred.unsqueeze(0).cpu().detach().numpy(),
                                occ_p = occ_pred.unsqueeze(0).cpu().detach().numpy(),
                                query_points = query_points.unsqueeze(0).cpu().detach().numpy(),
                                )
            
            
            
            # print('True_positive_occ:',test_metric['True_positive_occ'],'True_negative_occ:',test_metric['True_negative_occ'],'False_positive_occ:',test_metric['False_positive_occ'],'False_negative_occ:',test_metric['False_negative_occ'])
            
            
            if metrics['occlusion_accuracy'] > 0:
                occlusion_accuracy_list.append([metrics['occlusion_accuracy']][0][0])
            if metrics['pts_within_1'] > 0:
                pts_within_1_list.append([metrics['pts_within_1']][0][0])
            if metrics['pts_within_2'] > 0:
                pts_within_2_list.append([metrics['pts_within_2']][0][0])
            if metrics['pts_within_4'] > 0:
                pts_within_4_list.append([metrics['pts_within_4']][0][0])
            if metrics['pts_within_8'] > 0:
                pts_within_8_list.append([metrics['pts_within_8']][0][0])
            if metrics['pts_within_16'] > 0:
                pts_within_16_list.append([metrics['pts_within_16']][0][0])
            if metrics['jaccard_1'] > 0:
                jaccard_1_list.append([metrics['jaccard_1']][0][0])
            if metrics['jaccard_2'] > 0:
                jaccard_2_list.append([metrics['jaccard_2']][0][0])
            if metrics['jaccard_4'] > 0:
                jaccard_4_list.append([metrics['jaccard_4']][0][0])
            if metrics['jaccard_8'] > 0:
                jaccard_8_list.append([metrics['jaccard_8']][0][0])
            if metrics['jaccard_16'] > 0:
                jaccard_16_list.append([metrics['jaccard_16']][0][0])
            if metrics['average_jaccard'] > 0:
                average_jaccard_list.append([metrics['average_jaccard']][0][0])
                # print([metrics['average_jaccard']][0][0])
            if metrics['average_pts_within_thresh'] > 0:
                average_pts_within_thresh_list.append([metrics['average_pts_within_thresh']][0][0])
                
            # print('average_jaccard:',[metrics['average_jaccard']][0][0],"average_pts_within_thresh:",[metrics['average_pts_within_thresh']][0][0],"occlusion_accuracy:",[metrics['occlusion_accuracy']][0][0])
                
            # if test_metric['True_positive_occ'] > 0:
            True_positive_occ_list.append(test_metric['True_positive_occ'])
            # if test_metric['True_negative_occ'] > 0:
            True_negative_occ_list.append(test_metric['True_negative_occ'])
            # if test_metric['False_positive_occ'] > 0:
            False_positive_occ_list.append(test_metric['False_positive_occ'])
            # if test_metric['False_negative_occ'] > 0:
            False_negative_occ_list.append(test_metric['False_negative_occ'])
            

            True_positive_dist_list.append(test_metric['True_positive_dist'])

            True_negative_dist_list.append(test_metric['True_negative_dist'])

            False_positive_dist_list.append(test_metric['False_positive_dist'])

            False_negative_dist_list.append(test_metric['False_negative_dist'])
            
            True_positive_dist_time_list.append(test_metric['True_positive_dist_list'])

            True_negative_dist_time_list.append(test_metric['True_negative_dist_list'])

            False_positive_dist_time_list.append(test_metric['False_positive_dist_list'])

            False_negative_dist_time_list.append(test_metric['False_negative_dist_list'])
            
            True_positive_occ_time_list.append(test_metric['True_positive_occ_list'])

            True_negative_occ_time_list.append(test_metric['True_negative_occ_list'])

            False_positive_occ_time_list.append(test_metric['False_positive_occ_list'])

            False_negative_occ_time_list.append(test_metric['False_negative_occ_list'])
                
            # for i in range(data_process['target_points'].shape[1]):
            #     print(data_process['target_points'][0,i])
            #     Time.sleep(3)
                
            if not os.path.exists('{}_video'.format(datasets)):
                os.makedirs('{}_video'.format(datasets))
            
            if need_video == True: 
                # traj_pred = traj_pred.permute(0,2,1,3)
                for x in range(B):
                    
                    # traj_pred[occ_pred == True] = 0
                
                    # disp_gt = dataset.plot_tracks(data_process['video'][x].numpy()/255, data_process['target_points'][x].numpy(),
                    #         data_process['occluded'][x].numpy())
                    # media.write_video('{}_video/{}_gt.mp4'.format(datasets, i), disp_gt, fps=1)
                
                    # disp = dataset.plot_tracks(data_process['video'][x].numpy()/255, traj_pred[x].cpu().detach().numpy(),
                    #         occ_pred[x].cpu().detach().numpy())
                
                    # media.write_video('{}_video/{}.mp4'.format(datasets, i), disp, fps=1)
                    disp = plot_tracks(data_process['video'][x].numpy()/255, traj_pred.cpu().detach().numpy(),
                            occ_pred.cpu().detach().numpy(), data_process['target_points'][x].numpy(), data_process['occluded'][x].numpy())
                
                    media.write_video('{}_video/{}.mp4'.format(datasets, i), disp, fps=5)
                    i = i+1
                    
        #     data_list.append({
        #     'traj': traj_pred.cpu().detach().numpy(),
        #     'occ': occ_pred.cpu().detach().numpy(),
        #     })
            
        # with open("ours.pickle", 'wb') as f:
        #     pickle.dump(data_list, f)
                    
        print("Loss:",total_loss/len(TEST_dataloader))
                
        occlusion_accuracy = sum(occlusion_accuracy_list)/len(occlusion_accuracy_list)
        pts_within_1 = sum(pts_within_1_list)/len(pts_within_1_list)
        pts_within_2 = sum(pts_within_2_list)/len(pts_within_2_list)
        pts_within_4 = sum(pts_within_4_list)/len(pts_within_4_list)
        pts_within_8 = sum(pts_within_8_list)/len(pts_within_8_list)
        pts_within_16 = sum(pts_within_16_list)/len(pts_within_16_list)
        jaccard_1 = sum(jaccard_1_list)/len(jaccard_1_list)
        jaccard_2 = sum(jaccard_2_list)/len(jaccard_2_list)
        jaccard_4 = sum(jaccard_4_list)/len(jaccard_4_list)
        jaccard_8 = sum(jaccard_8_list)/len(jaccard_8_list)
        jaccard_16 = sum(jaccard_16_list)/len(jaccard_16_list)
        average_jaccard = sum(average_jaccard_list)/len(average_jaccard_list)
        average_pts_within_thresh = sum(average_pts_within_thresh_list)/len(average_pts_within_thresh_list)
            
        True_positive_occ = sum(True_positive_occ_list)/len(True_positive_occ_list)
        True_negative_occ = sum(True_negative_occ_list)/len(True_negative_occ_list)
        False_positive_occ = sum(False_positive_occ_list)/len(False_positive_occ_list)
        False_negative_occ = sum(False_negative_occ_list)/len(False_negative_occ_list)
        True_positive_dist = sum(True_positive_dist_list)/len(True_positive_dist_list)
        True_negative_dist = sum(True_negative_dist_list)/len(True_negative_dist_list)
        False_positive_dist = sum(False_positive_dist_list)/len(False_positive_dist_list)
        False_negative_dist = sum(False_negative_dist_list)/len(False_negative_dist_list)
        
        True_positive_dist_time = np.sum(np.stack(True_positive_dist_time_list, axis = 1), axis = 1 ,keepdims = False)/(np.sum(np.stack(True_positive_dist_time_list, axis = 1)!=0, axis = 1 ,keepdims = False)+1e-8)
        True_negative_dist_time = np.sum(np.stack(True_negative_dist_time_list, axis = 1), axis = 1 ,keepdims = False)/(np.sum(np.stack(True_negative_dist_time_list, axis = 1)!=0, axis = 1 ,keepdims = False)+1e-8)
        False_positive_dist_time = np.sum(np.stack(False_positive_dist_time_list, axis = 1), axis = 1 ,keepdims = False)/(np.sum(np.stack(False_positive_dist_time_list, axis = 1)!=0, axis = 1 ,keepdims = False)+1e-8)
        False_negative_dist_time = np.sum(np.stack(False_negative_dist_time_list, axis = 1), axis = 1 ,keepdims = False)/(np.sum(np.stack(False_negative_dist_time_list, axis = 1)!=0, axis = 1 ,keepdims = False)+1e-8)
        
        True_positive_occ_time = np.sum(np.stack(True_positive_occ_time_list, axis = 1), axis = 1 ,keepdims = False)/(np.sum(np.stack(True_positive_occ_time_list, axis = 1)!=0, axis = 1 ,keepdims = False)+1e-8)      
        True_negative_occ_time = np.sum(np.stack(True_negative_occ_time_list, axis = 1), axis = 1 ,keepdims = False)/(np.sum(np.stack(True_negative_occ_time_list, axis = 1)!=0, axis = 1 ,keepdims = False)+1e-8)
        False_positive_occ_time = np.sum(np.stack(False_positive_occ_time_list, axis = 1), axis = 1 ,keepdims = False)/(np.sum(np.stack(False_positive_occ_time_list, axis = 1)!=0, axis = 1 ,keepdims = False)+1e-8)
        False_negative_occ_time = np.sum(np.stack(False_negative_occ_time_list, axis = 1), axis = 1 ,keepdims = False)/(np.sum(np.stack(False_negative_occ_time_list, axis = 1)!=0, axis = 1 ,keepdims = False)+1e-8)
        
        # Time_index = np.arange(250)
        
        # plt.xlim(0,250)
        # plt.ylim(0,50)
        # plt.plot(Time_index, True_positive_dist_time)
        # plt.savefig('figs/True_positive_dist_time.png')
        # plt.clf()
        # plt.xlim(0,250)
        # plt.ylim(0,50)
        # plt.plot(Time_index, True_negative_dist_time)
        # plt.savefig('figs/True_negative_dist_time.png')
        # plt.clf()
        # plt.xlim(0,250)
        # plt.ylim(0,50)
        # plt.plot(Time_index, False_positive_dist_time)
        # plt.savefig('figs/False_positive_dist_time.png')
        # plt.clf()
        # plt.xlim(0,250)
        # plt.ylim(0,50)
        # plt.plot(Time_index, False_negative_dist_time)
        # plt.savefig('figs/False_negative_dist_time.png')
        # plt.clf()
        # plt.xlim(0,250)
        # plt.ylim(0,1)
        # plt.plot(Time_index, True_positive_occ_time)
        # plt.savefig('figs/True_positive_occ_time.png')
        # plt.clf()
        # plt.xlim(0,250)
        # plt.ylim(0,1)
        # plt.plot(Time_index, True_negative_occ_time)
        # plt.savefig('figs/True_negative_occ_time.png')
        # plt.clf()
        # plt.xlim(0,250)
        # plt.ylim(0,1)
        # plt.plot(Time_index, False_positive_occ_time)
        # plt.savefig('figs/False_positive_occ_time.png')
        # plt.clf()
        # plt.xlim(0,250)
        # plt.ylim(0,1)
        # plt.plot(Time_index, False_negative_occ_time)
        # plt.savefig('figs/False_negative_occ_time.png')
        # plt.clf()
        
        
        print("average_jaccard:",average_jaccard,"average_pts_within_thresh:",average_pts_within_thresh,"occlusion_accuracy:",occlusion_accuracy)
        print('True_positive_occ:',True_positive_occ,'True_negative_occ:',True_negative_occ,'False_positive_occ:',False_positive_occ,'False_negative_occ',False_negative_occ)
        print('True_positive_dist:',True_positive_dist,'True_negative_dist:',True_negative_dist,'False_positive_dist:',False_positive_dist,'False_negative_dist',False_negative_dist)
    
    

        
if __name__ == '__main__':
    test()