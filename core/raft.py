import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.update import BasicUpdateBlock, SmallUpdateBlock
from core.extractor import BasicEncoder, SmallEncoder
from core.corr import CorrBlock, AlternateCorrBlock
from core.utils.utils import bilinear_sampler, coords_grid, upflow8
from einops import rearrange, repeat
import torchvision.transforms as transforms
import core.blocks as blocks
from core.utils.utils import bilinear_sampler

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
        

class color_normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.prob_color_augment = 0.8
        self.prob_color_drop = 0.2
        self.prob_blur = 0.25
        
        
    def forward(self, video, is_train):
        
        video = video/255.0
        if is_train:
            color_augment = torch.rand(1)
            color_drop = torch.rand(1)
            color_blur = torch.rand(1)
            
            if color_augment < self.prob_color_augment:
                video = transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.4, hue= 0.2).forward(video)
                video = torch.clip(video, 0, 1)
            
            if color_drop < self.prob_color_drop:
                video = transforms.Grayscale(3)(video)
        
        video = video*255
        
        
        return video

        
    
class feature_fusion(nn.Module):
    def __init__(self, dim = 256):
        super().__init__()
        self.W_iz = nn.Conv2d(dim + 1, dim, 1)
        self.W_hz = nn.Conv2d(dim + 1, dim, 1)
        
    def forward(self, old_feature, gt_feature, time_index):
        old_feature_ = torch.cat([old_feature, time_index], dim = 1)
        gt_feature_ = torch.cat([gt_feature, time_index], dim = 1)
        
        z = torch.sigmoid(self.W_iz(old_feature_) + self.W_hz(gt_feature_))
        
        feature_out = (1-z)*old_feature + z*gt_feature
        
        return feature_out

    

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout, stride = args.stride)       
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout, stride = args.stride)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
            
        self.color_normalize = color_normalize()
        self.feature_fusion = feature_fusion()
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8, dim_feedforward = 1024, batch_first = True, norm_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6, norm = nn.LayerNorm(256))
        self.decoder_output = nn.Linear(256, 256)
        
        self.xent = nn.CrossEntropyLoss(reduction="none")
        self.temperature = 0.07
        self.edgedrop_rate = 0.1

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, args):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//args.stride, W//args.stride, device=img.device)
        coords1 = coords_grid(N, H//args.stride, W//args.stride, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask, args):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, args.stride, args.stride, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(args.stride * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, args.stride*H, args.stride*W)
    
    def upsample_occ(self, flow, mask, args):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, args.stride, args.stride, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, args.stride*H, args.stride*W)

    def forward(self, video, reference_frame, iters=4, flow_init=None, upsample=True, win_leng = 8, new_feature_map = None, dict = None, index = 0):
        """ Estimate optical flow between pair of frames """
        
        v_plus_p = torch.cat([video, reference_frame], dim = 1)
        v_plus_p = v_plus_p.squeeze(0) # T, C, H, W
        
        T, C, H, W = v_plus_p.shape
        
        T = T - 1

        v_plus_p = 2 * (v_plus_p / 255.0) - 1.0

        v_plus_p = v_plus_p.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap = self.fnet(v_plus_p)   
        
        fmap = fmap.float()
        fmap1 = fmap[-1:].repeat(T, 1, 1, 1)
        
        fmap2 = fmap[:-1]
        
        if dict == None:
            new_feature_map = fmap1[-1:].detach()
        else:
            old_feature = dict['old']
            gt_feature = fmap[-1:].detach()
            coords = dict['coord']
            rep = dict['rep']
            min_index = dict['min_index']
            mem = dict['mem']
            
            # add index
            time_index = torch.tensor([index], device = fmap.device)
            time_index = repeat(time_index, '() -> () () h w', h = H//self.args.stride, w = W//self.args.stride)
            
            # feature fusion
            decoder_input = self.feature_fusion(old_feature, mem, time_index) # 1, C, H, W
            
            r = 1
            dx = torch.linspace(-r, r, 2*r+1, device=fmap1.device)
            dy = torch.linspace(-r, r, 2*r+1, device=fmap1.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = rearrange(coords, 't c h w -> t (h w) () c')
            delta_lvl = delta.view(1, 1, -1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            feature_sample = bilinear_sampler(dict['new'], coords_lvl) # t c hw n
            feature_sample = feature_sample[min_index, :, torch.arange(H*W//(self.args.stride**2))]

            memory = rearrange(feature_sample, 'hw c n -> hw n c')
            
            tgt = rearrange(decoder_input, '() c h w -> (h w) () c')
            
            with autocast(enabled=self.args.mixed_precision):
                decoder_output = self.decoder_output(self.decoder(tgt, memory)) # (h w) () c
                
            new_feature_map = rearrange(decoder_output, '(h w) () c -> c h w', h = H//self.args.stride)*rep + dict['old']*~rep
            
            mem = mem*(1 - self.args.long_memory_lambda) + self.args.long_memory_lambda*rearrange(memory[:, ((2*r+1)**2-1)//2], '(h w) c -> c h w', h = H//self.args.stride)
            
            dict['mem'] = mem*rep + dict['mem']*~rep
            dict['mem'] = dict['mem'].detach()
            
            fmap1 = dict['mem'].repeat(T, 1, 1, 1)
            
        
        fmap1_new = new_feature_map.repeat(T, 1, 1, 1)
        fmap1_new = fmap1_new.float()
        
        
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            corr_fn_new = CorrBlock(fmap1_new, fmap2, radius=self.args.corr_radius)
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(v_plus_p)
            cnet = cnet[-1:].repeat(T, 1, 1, 1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(v_plus_p[:-1], self.args)
        
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        occ_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume # T, LRR, N_h, N_w
            corr_new = corr_fn_new(coords1)
            
            T, _, h, w = corr.shape
           
            corr = torch.cat([corr, corr_new], dim = 1)

            flow = coords1 - coords0
            
            if len(occ_predictions) != 0:
                occ_input = (occ_pred[:, :1] > 0.3).long()
            else:
                occ_input = torch.zeros(T, 1, h, w).long()
                
            
            with autocast(enabled=self.args.mixed_precision):

                net, flow_mask, occ_mask, delta_flow, occ = self.update_block(net, inp, corr, flow, occ_input = occ_input)
                
            occ = occ.float()
            
            occ_pred = torch.sigmoid(occ)
            
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if flow_mask is None:
                flow_up = upflow8(coords1 - coords0) # T, 2, N_h, N_w
            else:
                flow_up = self.upsample_flow(coords1 - coords0, flow_mask, self.args)
                
            flow_predictions.append(flow_up[:win_leng])
        
            occ  = self.upsample_occ(occ, occ_mask, self.args) # T, 2, H, W
            
            occ = torch.sigmoid(occ)
            
            occ_predictions.append(occ[:win_leng])
            
        flow_down = coords1 - coords0


        
        occ_p = occ_pred[:T//2, :1]
        uncertainty_p = occ_pred[:T//2, 1:]
        
        
        valid = (occ_p < 0.3) & (uncertainty_p < 0.8) # T//2, 1, H, W
        
        uncertain_update = uncertainty_p + ~valid*10000
        
        replace = torch.sum(valid, dim = 0, keepdim=False) > 0 # 1, H, W
        
        
        min_index = torch.argmin(uncertain_update, dim = 0, keepdim = True)
        
        min_index = rearrange(min_index, '() () h w -> (h w)')
            
        return flow_predictions, occ_predictions, flow_down[:win_leng], None, {'new':fmap2[:T//2].detach(), 'old':new_feature_map.detach(), "rep": replace.detach(), 'coord': coords1[:T//2].detach(), 'min_index': min_index.detach(), 'mem': fmap1[-1:].detach()}
