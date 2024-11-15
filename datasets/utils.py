import logging
import numpy as np
import os
import random
import time
from collections import defaultdict
import cv2
import torch
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms

from utils.envs import pathmgr
from utils.general import LOGGER

def loader_worker_init_fn(dataset):
    """
    Create init function passed to pytorch data loader.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
    """
    return None



def create_sampler(dataset, shuffle, cfg):
    """
    Create sampler for the given dataset.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
        shuffle (bool): set to ``True`` to have the data reshuffled
            at every epoch.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        sampler (Sampler): the created sampler.
    """
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    
    return sampler



def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    Sample frames among the corresponding clip.

    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames

    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    seq = list(range(center_idx- half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq


def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.

    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.

    Returns:
        imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = []
        for image_path in image_paths:
            with pathmgr.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8) # 转化成np数组
                img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
            imgs.append(img)

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            LOGGER.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry -1:
            raise Exception("Failed to load images {}".format(image_paths))
        


def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]   # RGB -> BGR

    # 快慢路径设置

    frame_list = [frames]

    return frame_list

def xyxy2xywhn(x, clip=False, eps=0.0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = ((x[..., 2] + x[..., 4]) / 2)  # x center
    y[..., 3] = ((x[..., 3] + x[..., 5]) / 2)  # y center
    y[..., 4] = (x[..., 4] - x[..., 2])  # width
    y[..., 5] = (x[..., 5] - x[..., 3])  # height
    return y

def xyxy2xywh(x):
    y = x.copy()
    y[0] = (x[0] + x[2]) / 2
    y[1] = (x[1] + x[3]) / 2
    y[2] = x[2] - x[0]
    y[3] = x[3] - x[1]
    return y
    
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y =x.clone() if isinstance(x, torch.Tensor) else np.copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y