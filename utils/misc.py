import json
import logging
import math
import numpy as np
import os
from datetime import datetime
import psutil
import torch
import torchvision.io as io
from fvcore.nn.activation_count import activation_count
from fvcore.nn.flop_count import flop_count
from matplotlib import pyplot as plt
from torch import nn
from torchvision.utils import make_grid

import utils.multiprocessing as mpu
from utils.general import LOGGER
from datasets.utils import pack_pathway_output

def launch_job(cfg,init_method ,func, daemon=False):
    """
    Run 'func' on GPU, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        func (function): job to run on GPU(s)
    """
    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs = cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                func,
                init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(cfg=cfg)

def frozen_bn_stats(model):
    """
    Set all the bn layers to eval mode.
    Args:
        model (model): model to set bn layers to eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()


def params_count(model, ignore_bn=False):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm3d):
                for p in m.parameters(recurse=False):
                    count += p.numel()
    return count


def _get_model_analysis_input(cfg, use_train_input):
    """
    Return a dummy input for model analysis with batch size 1. The input is
        used for analyzing the model (counting flops and activations etc.).
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        use_train_input (bool): if True, return the input for training. Otherwise,
            return the input for testing.

    Returns:
        inputs: the input for model analysis.
    """
    rgb_dimension = 3
    if use_train_input:
        if "imagenet" in cfg.TRAIN.DATASET:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.TRAIN_CROP_SIZE,
                cfg.DATA.TRAIN_CROP_SIZE,
            )
        else:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.NUM_FRAMES,
                cfg.DATA.TRAIN_CROP_SIZE,
                cfg.DATA.TRAIN_CROP_SIZE,
            )
    else:
        if "imagenet" in cfg.TEST.DATASET:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.TEST_CROP_SIZE,
                cfg.DATA.TEST_CROP_SIZE,
            )
        else:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.NUM_FRAMES,
                cfg.DATA.TEST_CROP_SIZE,
                cfg.DATA.TEST_CROP_SIZE,
            )
    model_inputs = pack_pathway_output(cfg, input_tensors)
    for i in range(len(model_inputs)):
        model_inputs[i] = model_inputs[i].unsqueeze(0)
        if cfg.NUM_GPUS:
            model_inputs[i] = model_inputs[i].cuda(non_blocking=True)

    # If detection is enabled, count flops for one proposal.
    if cfg.DETECTION.ENABLE:
        bbox = torch.tensor([[0, 0, 1.0, 0, 1.0]])
        if cfg.NUM_GPUS:
            bbox = bbox.cuda()
        inputs = (model_inputs, bbox)
    else:
        inputs = (model_inputs,)
    return inputs


def get_model_stats(model, cfg, mode, use_train_input):
    """
    Compute statistics for the current model given the config.
    Args:
        model (model): model to perform analysis.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        mode (str): Options include `flop` or `activation`. Compute either flop
            (gflops) or activation count (mega).
        use_train_input (bool): if True, compute statistics for training. Otherwise,
            compute statistics for testing.

    Returns:
        float: the total number of count of the given model.
    """
    assert mode in [
        "flop",
        "activation",
    ], "'{}' not supported for model analysis".format(mode)
    if mode == "flop":
        model_stats_fun = flop_count
    elif mode == "activation":
        model_stats_fun = activation_count

    # Set model to evaluation mode for analysis.
    # Evaluation mode can avoid getting stuck with sync batchnorm.
    model_mode = model.training
    model.eval()
    inputs = _get_model_analysis_input(cfg, use_train_input)
    count_dict, *_ = model_stats_fun(model, inputs)
    count = sum(count_dict.values())
    model.train(model_mode)
    return count



def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024**3

def log_model_info(model, cfg, use_train_input=True):
    """
    Log info, includes number of parameters, gpu usage, gflops and activation count.
        The model info is computed when the model is in validation mode.
    Args:
        model (model): model to log the info.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        use_train_input (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    LOGGER.info("Model:\n{}".format(model))
    params = params_count(model)
    LOGGER.info("Params: {:,}".format(params))
    LOGGER.info("Mem: {:,} MB".format(gpu_mem_usage()))
    flops = get_model_stats(model, cfg, "flop", use_train_input)
    LOGGER.info("Flops: {:,} G".format(flops))
    LOGGER.info(
        "Activations: {:,} M".format(
            get_model_stats(model, cfg, "activation", use_train_input)
        )
    )
    LOGGER.info("nvidia-smi")
    os.system("nvidia-smi")
    return flops, params


def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))
    

def is_eval_epoch(cfg, cur_epoch, multigrid_schedule):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (int): current epoch.
        multigrid_schedule (List): schedule for multigrid training.
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True
    if multigrid_schedule is not None:
        prev_epoch = 0
        for s in multigrid_schedule:
            if cur_epoch < s[-1]:
                period = max(
                    (s[-1] - prev_epoch) // cfg.MULTIGRID.EVAL_FREQ + 1, 1
                )
                return (s[-1] - 1 - cur_epoch) % period == 0
            prev_epoch = s[-1]
    
    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0
