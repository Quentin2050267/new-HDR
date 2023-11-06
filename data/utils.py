import os
import numpy as np
from imageio import imread, imsave
import re
import cv2
import torch
import torch.nn as nn
import random
random.seed(0)

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', 'npy']

        
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(dataroot):
    paths = None
    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))
    return  paths

def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def read_img(path):
    if os.path.splitext(path)[1] == '.npy':
        img = np.load(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is  None:
        # 图像成功加载
        # 进行后续处理
        # print('success{}'.format(path))
        # pass
        print('error{}'.format(path))
        pass
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def read_img_seq(img_paths):
    imgs = [read_img(v) for v in img_paths]
    return imgs

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
            kernel_size=2, stride=2, padding=0, bias=False,
            groups=in_channels)
    LH = net(in_channels, in_channels,
            kernel_size=2, stride=2, padding=0, bias=False,
            groups=in_channels)
    HL = net(in_channels, in_channels,
            kernel_size=2, stride=2, padding=0, bias=False,
            groups=in_channels)
    HH = net(in_channels, in_channels,
            kernel_size=2, stride=2, padding=0, bias=False,
            groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

def generate_mask(tensor, patch_size, mask_ratio):
    _, _, height, width = tensor.size()
    mask = torch.ones(height, width)
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches = num_patches_h * num_patches_w
    num_patches_to_mask = int(num_patches * mask_ratio)

    # Randomly select patches to mask
    patch_indices = torch.randperm(num_patches)[:num_patches_to_mask]
    patch_indices_h = torch.div(patch_indices, num_patches_w, rounding_mode='floor')
    patch_indices_w = patch_indices % num_patches_w

    # Apply mask to selected patches
    for j in range(num_patches_to_mask):
        mask[
            patch_indices_h[j] * patch_size: (patch_indices_h[j] + 1) * patch_size,
            patch_indices_w[j] * patch_size: (patch_indices_w[j] + 1) * patch_size
        ] = 0

    # Expand mask to match the shape of the input tensor
    mask = mask.unsqueeze(0).expand_as(tensor)
    return mask

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))
        
### IO Related ###
def make_file(f):
    if not os.path.exists(f):
        os.makedirs(f)
    #else:  raise Exception('Rendered image directory %s is already existed!!!' % directory)

def make_files(f_list):
    for f in f_list:
        make_file(f)

def empty_file(name):
    with open(name, 'w') as f:
        f.write(' ')

def read_list(list_path,ignore_head=False, sort=False):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists

def split_list(in_list, percent=0.99):
    num1 = int(len(in_list) * percent)
    #num2 = len(in_list) - num2
    rand_index = np.random.permutation(len(in_list))
    list1 = [in_list[l] for l in rand_index[:num1]]
    list2 = [in_list[l] for l in rand_index[num1:]]
    return list1, list2

def write_string(filename, string):
    with open(filename, 'w') as f:
        f.write('%s\n' % string)

def save_list(filename, out_list):
    f = open(filename, 'w')
    #f.write('#Created in %s\n' % str(datetime.datetime.now()))
    for l in out_list:
        f.write('%s\n' % l)
    f.close()

def create_dirs(root, dir_list, sub_dirs):
    for l in dir_list:
        makeFile(os.path.join(root, l))
        for sub_dir in sub_dirs:
            makeFile(os.path.join(root, l, sub_dir))

#### String Related #####
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def dict_to_string(dicts, start='\t', end='\n'):
    strs = '' 
    for k, v in sorted(dicts.items()):
        strs += '%s%s: %s%s' % (start, str(k), str(v), end) 
    return strs

def float_list_to_string(l):
    strs = ''
    for f in l:
        strs += ',%.2f' % (f)
    return strs

def insert_suffix(name_str, suffix):
    str_name, str_ext = os.path.splitext(name_str)
    return '%s_%s%s' % (str_name, suffix, str_ext)

def insert_char(mystring, position, chartoinsert):
    mystring = mystring[:position] + chartoinsert + mystring[position:] 
    return mystring  

def get_datetime(minutes=False):
    t = datetime.datetime.now()
    dt = ('%02d-%02d' % (t.month, t.day))
    if minutes:
        dt += '-%02d-%02d' % (t.hour, t.minute)
    return dt

def check_in_list(list1, list2):
    contains = []
    for l1 in list1:
        for l2 in list2:
            if l1 in l2.lower():
                contains.append(l1)
                break
    return contains

def remove_slash(string):
    if string[-1] == '/':
        string = string[:-1]
    return string

### Debug related ###
def check_div_by_2exp(h, w):
    num_h = np.log2(h)
    num_w = np.log2(w)
    if not (num_h).is_integer() or not (num_w).is_integer():
        raise Exception('Width or height cannot be devided exactly by 2exponet')
    return int(num_h), int(num_w)

def raise_not_defined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)

