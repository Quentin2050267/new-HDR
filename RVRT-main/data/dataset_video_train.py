import numpy as np
import random
import torch
from torch.nn import functional as F

from pathlib import Path
import torch.utils.data as data
# from basicsr.data.transforms import augment, paired_random_crop
# from basicsr.utils import img2tensor
import utils.utils_video as utils_video

# from basicsr.utils.options import *

import os

import data.utils as utils

    
class YouTubeDataset(data.Dataset):
    def __init__(self, opt):
        super(YouTubeDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        # 所有图片的绝对路径
        self.paths_GT = utils.get_image_paths(self.gt_root)
        self.paths_LQ = utils.get_image_paths(self.lq_root)
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))

        self.nframes = opt['num_frame']
        self._get_num()
        
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        
        
        self.load_pretrain = self.opt.get('load_pretrain', False)
        if self.load_pretrain:
            self.LL, self.LH, self.HL, self.HH = utils.get_wav(3, pool=True)
            self.iLL, self.iLH, self.iHL, self.iHH = utils.get_wav(3, pool=False)
        self.mask_ratio = self.opt.get('mask_ratio', False)
        self.random_resize = self.opt.get('random_resize', False)
        self.epoch = 0
        # self._size()
        
    def _get_num(self):
        self.frame_counts = {}
        for file in self.paths_GT:
            name = os.path.basename(file).split('_')[0]
            if name in self.frame_counts:
                self.frame_counts[name] += 1
            else:
                self.frame_counts[name] = 1
    
    # def _set_epoch(self, epoch):
    #     self.epoch = epoch

    # def _size(self):
    #     img_gt_path = self.paths_GT[0]
    #     img_lq_path = self.paths_LQ[0]
    #     img_lq = utils.read_img(img_lq_path)
    #     img_gt= utils.read_img(img_gt_path)
    #     h_lq, w_lq = img_lq.size()[-2:]
    #     h_gt, w_gt = img_gt.size()[-2:]
    #     self.h_lq=h_lq
    #     self.h_gt=h_gt
    #     self.w_lq=w_lq
    #     self.w_gt=w_gt
        
    def __getitem__(self, index):
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        img_gt_path = self.paths_GT[index]
        img_lq_path = self.paths_LQ[index]

        key = os.path.basename(img_gt_path).split('.')[0]
        clip_name, frame_name = key.split('_')  # key example: black_001
        
        # frame count of one clip
        self.count = self.frame_counts[clip_name]   
             
        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > self.count - self.nframes * interval:
            start_frame_idx = random.randint(1, self.count - self.nframes * interval)
        end_frame_idx = start_frame_idx + self.nframes * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            img_lq_path = f'{self.lq_root}/{clip_name}_{neighbor:03d}.png'
            img_gt_path = f'{self.gt_root}/{clip_name}_{neighbor:03d}.png'
            # print(img_lq_path)
            # get LQ
            img_lq = utils.read_img(img_lq_path)
            img_lqs.append(img_lq)

            # get GT
            img_gt = utils.read_img(img_gt_path)
            img_gts.append(img_gt)
  
        # downsample
        # img_gts=np.array(img_gts)
        # img_lqs=np.array(img_lqs)
        ###  3840*2160  ==>  960*540  ==>  320/384
        img_gts = F.interpolate(torch.from_numpy(img_gts).permute(0,3,1,2), size=(720,1280), mode='bilinear', align_corners=False)
        img_lqs = F.interpolate(torch.from_numpy(img_lqs).permute(0,3,1,2), size=(720,1280), mode='bilinear', align_corners=False)

        img_gts = [img_gts[i].permute(1,2,0).numpy() for i in range(img_gts.shape[0])]
        img_lqs = [img_lqs[i].permute(1,2,0).numpy() for i in range(img_lqs.shape[0])]

        # randomly crop
        img_gts, img_lqs = utils_video.paired_random_crop(img_gts, img_gts, gt_size, scale, img_gt_path)

        img_lqs.extend(img_gts)

        # augmentation - flip, rotate
        img_results = utils_video.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = utils_video.img2tensor(img_results)
        

        # img_gts, img_lqs, crops_info = paired_random_crop(img_gts, img_lqs, gt_size, scale, self.epoch, img_gt_path, True)
        # img_lqs.extend(img_gts) 
        # # augmentation - flip, rotate
        # img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        # img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)
        
        if self.load_pretrain:
            # 将sdr作为gt
            img_gts = img_lqs  # [0-1]
            # 多级wavelet
            ldr_LL = img_gts * 255  # [0-255]
            img_lqs = torch.zeros_like(img_gts)
            
            for i in range(self.load_pretrain):
                ldr_LH = self.LH(ldr_LL)
                ldr_HL = self.HL(ldr_LL)
                ldr_HH = self.HH(ldr_LL)
                ldr_LL = self.LL(ldr_LL)

                ldr_iLL = self.iLL(ldr_LL)
                ldr_iLH = self.iLH(ldr_LH)
                ldr_iHL = self.iHL(ldr_HL)
                ldr_iHH = self.iHH(ldr_HH)
                
                for j in range(i):
                    ldr_iLL = self.iLL(ldr_iLL)
                    ldr_iLH = self.iLL(ldr_iLH)
                    ldr_iHL = self.iLL(ldr_iHL)
                    ldr_iHH = self.iLL(ldr_iHH)
                
            # 保留高频
            # img_lqs += ldr_iLH.to(torch.uint8) + ldr_iHL.to(torch.uint8) + ldr_iHH.to(torch.uint8)
            # 保留低频
            img_lqs += ldr_iLL.to(torch.uint8)

            # 只学低频信息，意味着lq为高频,gt为ldr
            img_lqs /= 255.   # [0-1]
            img_lqs.to(torch.float32)
        
        # if self.mask_ratio:
        #     mask = utils.generate_mask(img_lqs, 16, self.mask_ratio)
        #     img_lqs = img_lqs * mask

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        # return {'lq': img_lqs, 'gt': img_gts, 'key': key}
        return {'L': img_lqs, 'H': img_gts, 'folder': clip_name, 'count': self.count} # , 'info': crops_info}

    def __len__(self):
        return len(self.paths_GT) 



if __name__ == '__main__':
    import cv2
    from imageio import imread, imsave
    import matplotlib.pyplot as plt
    from PIL import Image
    from basicsr.utils.options import *



    opt_path = r'/home/qintian/HDR-main/options/youtube.yml'
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = YouTubeDataset(dataset_opt)
            train_set._set_epoch(6)
            data = train_set[6]
            print(train_set.epoch)
            # print(train_set.paths_GT)
            lq=data['info']
            print(lq)

            #     gt = data['gt'][i].numpy().transpose(1, 2, 0)
            #     plt.subplot(2, 7, i+1)
            #     plt.imshow(lq)
            #     # plt.subplot(2, 7, 8+i)
            #     # plt.imshow(gt)
    
            # plt.show()
            

