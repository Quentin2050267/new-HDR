import sys
import os.path
import math
import argparse
import time
import random
import cv2
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import argparse
import os
import torch
import requests
import numpy as np
from os import path as osp
from collections import OrderedDict
from torch.utils.data import DataLoader

# from models.network_rvrt import RVRT as net
from utils import utils_image as util
# from data.dataset_video_test import VideoRecurrentTestDataset, VideoTestVimeo90KDataset, SingleVideoRecurrentTestDataset
from piq import srsim, ssim, psnr

'''
# --------------------------------------------
# training code for VRT/RVRT
# --------------------------------------------
'''


def main(json_path='options/rvrt/001_train_rvrt_videohdr_youtube.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))


    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G',
                                                           pretrained_path=opt['path']['pretrained_netG'])
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E',
                                                           pretrained_path=opt['path']['pretrained_netE'])
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],
                                                                             net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'],
                                                   drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # 检查输入，检查完后下面7行就可以删了,然后正式开始训练
            save_dir = opt['path']['images']
            util.mkdir(save_dir)
            for i in range(model.H.size(0)):
                gt = util.tensor2img(model.H[i], np.uint16)  # CHW-RGB to HCW-BGR uint16
                lq = util.tensor2img(model.L[i], np.uint8)  # CHW-RGB to HCW-BGR uint8
                cv2.imwrite(f'{save_dir}/{current_step:d}_{i}.png', gt)
                cv2.imwrite(f'{save_dir}/{current_step:d}_{i}.png', lq)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                          model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            if opt['use_static_graph'] and (current_step == opt['train']['fix_iter'] - 1):
                current_step += 1
                model.update_learning_rate(current_step)
                model.save(current_step)
                current_step -= 1
                logger.info('Saving models ahead of time when changing the computation graph with use_static_graph=True'
                            ' (we need it due to a bug with use_checkpoint=True in distributed training). The training '
                            'will be terminated by PyTorch in the next iteration. Just resume training with the same '
                            '.json config file.')

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                test_results = OrderedDict()
                test_results['psnr'] = []
                test_results['ssim'] = []
                test_results['deltaITP'] = []
                test_results['SRSIM'] = []

                for data in test_loader:
                    need_GT = False if test_loader.dataset.opt['dataroot_gt'] is None else True
                    model.feed_data(data, need_GT=need_GT)
                    img_name = osp.splitext(osp.basename(data['GT_path'][0]))[0]

                    model.test()
                    visuals = model.current_visuals(need_GT=need_GT)

                    sr_img = util.tensor2img(visuals['E'][0], np.uint16)  # CHW-RGB to HCW-BGR uint16

                    # save images
                    if opt['val']['save_img']:
                        save_dir = opt['path']['images']
                        util.mkdir(save_dir)
                        cv2.imwrite(f'{save_dir}/{img_name}_{current_step:d}.png', sr_img)


                    # calculate metric
                    gt_img = util.tensor2img(visuals['H'][0], np.uint16) # uint8 / uint16

                    img_GT = gt_img.astype(np.int32)
                    img = sr_img.astype(np.int32)

                    tensor_GT = torch.Tensor(img_GT).unsqueeze(0).permute(0, 3, 1, 2)
                    tensor_pred = torch.Tensor(img).unsqueeze(0).permute(0, 3, 1, 2)
                    SRSIM = srsim(tensor_GT, tensor_pred, data_range=65535.)
                    test_results['SRSIM'].append(SRSIM)

                    SSIM = ssim(tensor_GT, tensor_pred, data_range=65535., downsample=False)
                    test_results['ssim'].append(SSIM)

                    PSNR = psnr(tensor_GT, tensor_pred, data_range=65535.)
                    test_results['ssim'].append(PSNR)

                    deltaITP = util.calculate_hdr_deltaITP_released(img, img_GT)
                    test_results['deltaITP'].append(deltaITP)
    
                    print('{:10s} - PSNR: {:.6f} dB;  SSIM: {:.6f};  deltaITP: {:.6f};  SR-SIM: {:.6f}'.format(
                        img_name, PSNR, SSIM, deltaITP, SRSIM))

                ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
                ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
                ave_deltaITP = sum(test_results['deltaITP']) / len(test_results['deltaITP'])
                ave_srsim = sum(test_results['SRSIM']) / len(test_results['SRSIM'])
                logger.info('<epoch:{:3d}, iter:{:8,d} Average PSNR: {:.6f} dB; SSIM: {:.6f}; '
                            'deltaITP: {:.6f} dB; SR-SIM: {:.6f}'.format(
                    epoch, current_step, ave_psnr, ave_ssim, ave_deltaITP, ave_srsim))

            if current_step > opt['train']['total_iter']:
                logger.info('Finish training.')
                model.save(current_step)
                sys.exit()

if __name__ == '__main__':
    main()