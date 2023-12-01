# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

import imageio
import numpy as np

from typing import Any, Optional
from torchvision.transforms import ColorJitter, GaussianBlur
from PIL import Image
import cv2

from numpy import random
import torch
import numpy as np
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import glob
import cv2
from torchvision.transforms import ColorJitter, GaussianBlur
import time

class CoTrackerData:
    """
    Dataclass for storing video tracks data.
    """

    video: torch.Tensor  # B, S, C, H, W
    segmentation: torch.Tensor  # B, S, 1, H, W
    trajectory: torch.Tensor  # B, S, N, 2
    visibility: torch.Tensor  # B, S, N
    # optional data
    valid: Optional[torch.Tensor] = None  # B, S, N
    seq_name: Optional[str] = None
    query_points: Optional[torch.Tensor] = None  # TapVID evaluation format


class CoTrackerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        sample_vis_1st_frame=False,
        use_augs=False,
    ):
        super(CoTrackerDataset, self).__init__()
        np.random.seed(0)
        torch.manual_seed(0)
        self.data_root = data_root
        self.seq_len = seq_len
        self.traj_per_sample = traj_per_sample
        self.sample_vis_1st_frame = sample_vis_1st_frame
        self.use_augs = use_augs
        self.crop_size = crop_size

        # photometric augmentation
        self.photo_aug = ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14
        )
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))

        self.blur_aug_prob = 0.25
        self.color_aug_prob = 0.25

        # occlusion augmentation
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [2, 100]
        self.eraser_max = 10

        # occlusion augmentation
        self.replace_aug_prob = 0.5
        self.replace_bounds = [2, 100]
        self.replace_max = 10

        # spatial augmentations
        self.pad_bounds = [0, 100]
        self.crop_size = crop_size
        self.resize_lim = [0.25, 2.0]  # sample resizes from here
        self.resize_delta = 0.2
        self.max_crop_offset = 50

        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5
        self.t_flip_prob = 0.5

    def getitem_helper(self, index):
        return NotImplementedError

    def __getitem__(self, index):
        gotit = False

        sample, gotit = self.getitem_helper(index)
        if not gotit:
            print("warning: sampling failed")
            # fake sample, so we can still collate
            sample = {
                'video': torch.zeros((self.S, 3, self.crop_size[0], self.crop_size[1])),
                'target_points': torch.zeros((self.S, self.N, 2)),
                'occluded': torch.ones((self.S, self.N)),
                'valids': torch.zeros((self.S, self.N)),
            }

        return sample, gotit

    def add_photometric_augs(self, rgbs, trajs, visibles, eraser=True, replace=True):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        if eraser:
            ############ eraser transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.eraser_max + 1)
                    ):  # number of times to occlude

                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(
                            self.eraser_bounds[0], self.eraser_bounds[1]
                        )
                        dy = np.random.randint(
                            self.eraser_bounds[0], self.eraser_bounds[1]
                        )
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        mean_color = np.mean(
                            rgbs[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0
                        )
                        rgbs[i][y0:y1, x0:x1, :] = mean_color

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        if replace:

            rgbs_alt = [
                np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8)
                for rgb in rgbs
            ]
            rgbs_alt = [
                np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8)
                for rgb in rgbs_alt
            ]

            ############ replace transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.replace_max + 1)
                    ):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(
                            self.replace_bounds[0], self.replace_bounds[1]
                        )
                        dy = np.random.randint(
                            self.replace_bounds[0], self.replace_bounds[1]
                        )
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        wid = x1 - x0
                        hei = y1 - y0
                        y00 = np.random.randint(0, H - hei)
                        x00 = np.random.randint(0, W - wid)
                        fr = np.random.randint(0, S)
                        rep = rgbs_alt[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                        rgbs[i][y0:y1, x0:x1, :] = rep

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        ############ photometric augmentation ############
        # if np.random.rand() < self.color_aug_prob:
        #     # random per-frame amount of aug
        #     print(Image.fromarray(rgbs[0]))
        #     rgbs = [
        #         np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8)
        #         for rgb in rgbs
        #     ]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [
                np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8)
                for rgb in rgbs
            ]

        return rgbs, trajs, visibles

    def add_spatial_augs(self, rgbs, trajs, visibles):
        T, N, __ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        rgbs = [rgb.astype(np.float32) for rgb in rgbs]

        ############ spatial transform ############

        # padding
        pad_x0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])

        rgbs = [
            np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs
        ]
        trajs[:, :, 0] += pad_x0
        trajs[:, :, 1] += pad_y0
        H, W = rgbs[0].shape[:2]

        # scaling + stretching
        scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
        scale_x = scale
        scale_y = scale
        H_new = H
        W_new = W

        scale_delta_x = 0.0
        scale_delta_y = 0.0

        rgbs_scaled = []
        for s in range(S):
            if s == 1:
                scale_delta_x = np.random.uniform(-self.resize_delta, self.resize_delta)
                scale_delta_y = np.random.uniform(-self.resize_delta, self.resize_delta)
            elif s > 1:
                scale_delta_x = (
                    scale_delta_x * 0.8
                    + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                )
                scale_delta_y = (
                    scale_delta_y * 0.8
                    + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                )
            scale_x = scale_x + scale_delta_x
            scale_y = scale_y + scale_delta_y

            # bring h/w closer
            scale_xy = (scale_x + scale_y) * 0.5
            scale_x = scale_x * 0.5 + scale_xy * 0.5
            scale_y = scale_y * 0.5 + scale_xy * 0.5

            # don't get too crazy
            scale_x = np.clip(scale_x, 0.2, 2.0)
            scale_y = np.clip(scale_y, 0.2, 2.0)

            H_new = int(H * scale_y)
            W_new = int(W * scale_x)

            # make it at least slightly bigger than the crop area,
            # so that the random cropping can add diversity
            H_new = np.clip(H_new, self.crop_size[0] + 10, None)
            W_new = np.clip(W_new, self.crop_size[1] + 10, None)
            # recompute scale in case we clipped
            scale_x = W_new / float(W)
            scale_y = H_new / float(H)

            rgbs_scaled.append(
                cv2.resize(rgbs[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR)
            )
            trajs[s, :, 0] *= scale_x
            trajs[s, :, 1] *= scale_y
        rgbs = rgbs_scaled

        ok_inds = visibles[0, :] > 0
        vis_trajs = trajs[:, ok_inds]  # S,?,2

        if vis_trajs.shape[1] > 0:
            mid_x = np.mean(vis_trajs[0, :, 0])
            mid_y = np.mean(vis_trajs[0, :, 1])
        else:
            mid_y = self.crop_size[0]
            mid_x = self.crop_size[1]

        x0 = int(mid_x - self.crop_size[1] // 2)
        y0 = int(mid_y - self.crop_size[0] // 2)

        offset_x = 0
        offset_y = 0

        for s in range(S):
            # on each frame, shift a bit more
            if s == 1:
                offset_x = np.random.randint(
                    -self.max_crop_offset, self.max_crop_offset
                )
                offset_y = np.random.randint(
                    -self.max_crop_offset, self.max_crop_offset
                )
            elif s > 1:
                offset_x = int(
                    offset_x * 0.8
                    + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
                    * 0.2
                )
                offset_y = int(
                    offset_y * 0.8
                    + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
                    * 0.2
                )
            x0 = x0 + offset_x
            y0 = y0 + offset_y

            H_new, W_new = rgbs[s].shape[:2]
            if H_new == self.crop_size[0]:
                y0 = 0
            else:
                y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)

            if W_new == self.crop_size[1]:
                x0 = 0
            else:
                x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)

            rgbs[s] = rgbs[s][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            trajs[s, :, 0] -= x0
            trajs[s, :, 1] -= y0

        H_new = self.crop_size[0]
        W_new = self.crop_size[1]

        # flip
        h_flipped = False
        v_flipped = False
        t_flipped = False
        if self.do_flip:
            # h flip
            if np.random.rand() < self.h_flip_prob:
                h_flipped = True
                rgbs = [rgb[:, ::-1] for rgb in rgbs]
            # v flip
            if np.random.rand() < self.v_flip_prob:
                v_flipped = True
                rgbs = [rgb[::-1] for rgb in rgbs]
                
            if np.random.rand() < self.t_flip_prob:
                t_flipped = True
                rgbs = rgbs[::-1].copy()
                
        if h_flipped:
            trajs[:, :, 0] = W_new - trajs[:, :, 0]
        if v_flipped:
            trajs[:, :, 1] = H_new - trajs[:, :, 1]
        if t_flipped:
            trajs = trajs[::-1].copy()
            visibles = visibles[::-1].copy()

        return rgbs, trajs, visibles

    def crop(self, rgbs, trajs):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        ############ spatial transform ############

        H_new = H
        W_new = W

        # simple random crop
        y0 = (
            0
            if self.crop_size[0] >= H_new
            else np.random.randint(0, H_new - self.crop_size[0])
        )
        x0 = (
            0
            if self.crop_size[1] >= W_new
            else np.random.randint(0, W_new - self.crop_size[1])
        )
        rgbs = [
            rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            for rgb in rgbs
        ]

        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0

        return rgbs, trajs


class PointOdysseyDataset(CoTrackerDataset):
    def __init__(
        self,
        dataset_location='/GPFS/public/xiongyicai/pointodyssey',
        dset='TRAIN',
        crop_size=(384, 512),
        seq_len=100,
        traj_per_sample=256,
        sample_vis_1st_frame=False,
        use_augs=False,
    ):
        super(PointOdysseyDataset, self).__init__(
            data_root=dataset_location,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            sample_vis_1st_frame=sample_vis_1st_frame,
            use_augs=use_augs,
        )

        self.pad_bounds = [0, 25]
        self.resize_lim = [0.75, 1.25]  # sample resizes from here
        self.resize_delta = 0.05
        self.max_crop_offset = 15
        print('loading pointodyssey dataset...')

        self.S = seq_len
        self.N = traj_per_sample

        self.use_augs = use_augs
        self.dset = dset

        self.rgb_paths = []
        self.traj_paths = []
        self.annotation_paths = []
        self.start_idx = []

        self.subdirs = []
        self.sequences = []
        self.seq_names = []
        if dset == "TRAIN":
            self.subdirs.append(os.path.join(dataset_location, 'train'))
        elif dset == "VAL":
            self.subdirs.append(os.path.join(dataset_location, 'val'))
        elif dset == "TEST":
            self.subdirs.append(os.path.join(dataset_location, 'test_clean'))

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*")):
                seq_name = seq.split('/')[-1]
                self.sequences.append(seq)
                self.seq_names.append(seq_name)

        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        
        ## load trajectories
        print('loading trajectories...')
        for seq in self.sequences:
            dir_path = dataset_location
            rgb_path = os.path.join(seq, 'rgbs')

            for ii in range(len(os.listdir(rgb_path)) - self.S):
                self.rgb_paths.append([os.path.join(dir_path, seq, 'rgbs', 'rgb_%05d.jpg' % (ii + jj + 1)) for jj in range(self.S)])
                self.annotation_paths.append(os.path.join(seq, 'annotations.npz'))
                self.start_idx.append(ii)

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))
        

    def getitem_helper(self, index):
        gotit = True
        # seq_name = self.seq_names[index]

        # npy_path = os.path.join(self.data_root, seq_name, seq_name + ".npy")
        # rgb_path = os.path.join(self.data_root, seq_name, "frames")

        # img_paths = sorted(os.listdir(rgb_path))
        # rgbs = []
        # for i, img_path in enumerate(img_paths):
        #     rgbs.append(imageio.v2.imread(os.path.join(rgb_path, img_path)))

        # rgbs = np.stack(rgbs)
        # annot_dict = np.load(npy_path, allow_pickle=True).item()
        # traj_2d = annot_dict["coords"]
        # visibility = annot_dict["visibility"]
        
        rgb_paths = self.rgb_paths[index]
        # print('rgb_paths', len(rgb_paths))

        full_idx = self.start_idx[index] + np.arange(self.S)
        annotations_path = self.annotation_paths[index]
        annotations = np.load(annotations_path, allow_pickle=True)
        # print(annotations.files)
        traj_2d = annotations['trajs_2d'][full_idx].astype(np.float32).transpose(1,0,2) # N, T, 2

        visibs = annotations['visibilities'][full_idx].astype(np.float32).transpose(1,0)
        
        # print(np.max(visibs))
        # print(np.min(visibs))
        visibility = (visibs==1).astype(np.float32)
        
        
        traj_isnan = (traj_2d == np.inf) + (traj_2d == -np.inf)
        valid_nan = np.sum(traj_isnan, axis = 1)
        valid_nan = (np.sum(valid_nan, axis = 1) == 0)
        traj_2d = traj_2d[valid_nan]
        visibility = visibility[valid_nan] # N, T
        
        valid_shrink = np.sum((visibility[:, :-1] != visibility[:, 1:]), axis = 1)
        
        valid_shrink = valid_shrink < self.S/15
        
        traj_2d = traj_2d[valid_shrink]
        visibility = visibility[valid_shrink] # N, T
        
        
        
        
        rgbs = []
        for rgb_path in rgb_paths:
            with Image.open(rgb_path) as im:
                rgbs.append(np.array(im)[:, :, :3])
                
        rgbs = np.stack(rgbs)

        # random crop
        assert self.seq_len <= len(rgbs)
        if self.seq_len < len(rgbs):
            start_ind = np.random.choice(len(rgbs) - self.seq_len, 1)[0]

            rgbs = rgbs[start_ind : start_ind + self.seq_len]
            traj_2d = traj_2d[:, start_ind : start_ind + self.seq_len]
            visibility = visibility[:, start_ind : start_ind + self.seq_len]

        traj_2d = np.transpose(traj_2d, (1, 0, 2))
        visibility = np.transpose(visibility, (1, 0))
        if self.use_augs:
            rgbs, traj_2d, visibility = self.add_photometric_augs(
                rgbs, traj_2d, visibility
            )
            rgbs, traj_2d, visibility = self.add_spatial_augs(rgbs, traj_2d, visibility)
        else:
            rgbs, traj_2d = self.crop(rgbs, traj_2d)

        visibility[traj_2d[:, :, 0] > self.crop_size[1] - 1] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > self.crop_size[0] - 1] = False
        visibility[traj_2d[:, :, 1] < 0] = False

        visibility = torch.from_numpy(visibility)
        traj_2d = torch.from_numpy(traj_2d)

        # visibile_pts_first_frame_inds = (visibility[0]).nonzero(as_tuple=False)[:, 0]

        # if self.sample_vis_1st_frame:
        #     visibile_pts_inds = visibile_pts_first_frame_inds
        # else:
        #     visibile_pts_mid_frame_inds = (visibility[self.seq_len // 2]).nonzero(
        #         as_tuple=False
        #     )[:, 0]
        #     visibile_pts_inds = torch.cat(
        #         (visibile_pts_first_frame_inds, visibile_pts_mid_frame_inds), dim=0
        #     )
        valid = torch.sum(visibility, dim = 0) > 0
        visibility = visibility[:, valid]
        traj_2d = traj_2d[:, valid]
        point_inds = torch.randperm(visibility.shape[1])[: self.traj_per_sample]
        if len(point_inds) < self.traj_per_sample:
            gotit = True


        trajs = traj_2d[:, point_inds].float()
        
        
        visibles = visibility[:, point_inds]
        valids = torch.ones((self.seq_len, self.traj_per_sample))

        rgbs = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float()
        segs = torch.ones((self.seq_len, 1, self.crop_size[0], self.crop_size[1]))
        sample = {
            'video': rgbs,
            'target_points': trajs,
            'occluded': 1-visibles,
            'valids': valids,
        }

        
        return sample, gotit

    def __len__(self):
        return len(self.rgb_paths)
    
    
if __name__ == '__main__':
    train_dataset = PointOdysseyDataset(dset='TRAIN',
                                        use_augs=True,
                                        seq_len=100,
                                        traj_per_sample=10000,
                                        crop_size=(384,512),)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=12,
        drop_last = True,
        pin_memory=True,
        )
    

    for data in train_dataloader:
        time.sleep(0.1)