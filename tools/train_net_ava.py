import math
import numpy as np
import pprint
import torch
import sys
import os
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from utils.general import LOGGER
from models import build_model
from pathlib import Path
from threading import Thread
from tqdm import tqdm
from copy import deepcopy
import torch.nn as nn
import cv2

import utils.distributed as du
import models.optimizer as optim
import utils.checkpoint as cu
from datasets import loader
from datasets.mixup import MixUp
from utils.meters import AVAMeter, EpochTimer
from utils.multigrid import MultigridSchedule
import utils.misc as misc
from models.losses_ava import ComputeLoss
from models.contrastive import contrastive_parameter_surgery
from datasets.utils import xyxy2xywhn, xywh2xyxy
from utils.general import non_max_suppression, scale_boxes, increment_path, colorstr, TQDM_BAR_FORMAT
from utils.metrics import ConfusionMatrix, box_iou, ap_per_class, fitness
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, de_parallel
from utils.autoanchor import check_anchors

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
# RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

def save_one_json(predn, jdict, path, class_map):
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    cur_epoch,
    cfg,
    epochs,
    compute_loss,
    writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()

    # train_meter.iter_tic()
    data_size = len(train_loader)

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL_PARA.CLASSES,
        )
    
    if cfg.MODEL_PARA.FROZEN_BN:
        misc.frozen_bn_stats(model)
    nb = len(train_loader)
    
    mloss = torch.zeros(3)
    # Explicitly declare reduction to mean.
    pbar = enumerate(train_loader)
    if du.is_master_proc():
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
    if du.is_master_proc():
        pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
    for cur_iter, (inputs, targets, index, time, meta) in pbar:
        targets = xyxy2xywhn(targets)
        targets = targets.cuda(non_blocking=True)
        mloss = mloss.cuda(non_blocking=True)
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # if not isinstance(labels, list):
            #     labels = labels.cuda(non_blocking=True)
            #     index = index.cuda(non_blocking=True)
            #     time = time.cuda(non_blocking=True)
            for key, val in meta.items():
                if key == "shapes" or key == "path" or key == "label_map":
                    continue
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        label_map = {}
        for l_map in meta["label_map"]:
            label_map.update(l_map)
        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )
        # Update the learning rate.
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)

        # train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples
        
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            # Explicitly declare reduction to mean.
            perform_backward = True
            optimizer.zero_grad()
            preds = model(inputs[0])
            loss, loss_items = compute_loss(preds, targets, label_map)

        # check Nan Loss
        misc.check_nan_losses(loss)
        if perform_backward:
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Clip grdients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            grad_norm = torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        else:
            grad_norm = optim.get_grad_norm_(model.parameters())

        # Update the parameters. (defaults to True)
        model, update_param = contrastive_parameter_surgery(
            model, cfg, epoch_exact, cur_iter
        )
        if update_param:
            scaler.step(optimizer)
        scaler.update()
        # if cfg.NUM_GPUS > 1:
        #     loss = du.all_reduce([loss])[0]
        # loss = loss.item()
        # Log
        if du.is_master_proc():
            mloss = (mloss * cur_iter + loss_items) / (cur_iter + 1)
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{cur_epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], inputs[0].shape[-1]))

        torch.cuda.synchronize()

    del inputs

    torch.cuda.empty_cache()

def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """
    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs
    
    update_bn_stats(model, _gen_loader(), num_iters)

@torch.no_grad()
def eval_epoch(
        val_loader, model, cur_epoch, cfg, train_loader, compute_loss,verbose=False, task='val',
):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    training = model is not None
    model.eval()
    device = next(model.parameters()).device # device
    # val_meter.iter_tic() # start time
    iouv = torch.linspace(0.5, 0.95, 10, device=device) # iou vector for mAP@0.5:0.95 [0.5000, 0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500, 0.9000, 0.9500]
    niou = iouv.numel() # number of iouv
    seen = 0 # initial
    names = dict(model.names.items() if hasattr(model, 'names') else model.module.names.items())

    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Directories
    save_dir = increment_path(Path(cfg.SAVE.PROJECT) / cfg.SAVE.NAME, exist_ok=cfg.SAVE.EXIST_OK)  # increment run
    (save_dir / 'labels' if cfg.SAVE.SAVE_TXT else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    confusion_matrix = ConfusionMatrix(nc=cfg.MODEL_PARA.CLASSES) # Confusion matrix initial
    loss = torch.zeros(3, device=device) # loss for 3 layer
    jdict, stats, ap, ap_class = [], [], [], [] # initial
    pbar = enumerate(val_loader)
    if du.is_master_proc():
        pbar = tqdm(pbar, desc=s, total=len(val_loader), bar_format=TQDM_BAR_FORMAT)
    for cur_iter, (inputs, targets, index, time, meta) in pbar:
        # boxes = xyxy2xywhn(meta["boxes"]) # xywh AVA使用
        targets = xyxy2xywhn(targets)
        targets = targets.cuda(non_blocking=True)
        nb, _, _, height, width = inputs[0].shape
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            for key, val in meta.items():
                if key == "shapes" or key == "path" or key == "label_map":
                    continue
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
            index = index.cuda()
            time = time.cuda()
        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )
        # val_meter.data_toc()
            
        if cfg.DETECTION.ENABLE:
            # Compute the predictions
            preds, train_out = model(inputs[0])
            
            # if cfg.VAL.COMPUTE_LOSS:
            #     loss += compute_loss(train_out, targets)[1]
            
            # NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if cfg.VAL.SAVE_HYBRID else []  # for autolabelling

            preds = non_max_suppression(preds,
                                        cfg.VAL.CONF_THRES,
                                        cfg.VAL.IOU_THRES,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=cfg.VAL.SINGLE_CLS)
            # Statistics per image
            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class

                # path, shape
                path, shape = Path(meta["path"][si]), meta["shapes"][si][0]
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                if cfg.VAL.SINGLE_CLS:
                    pred[:, 5] = 0
                predn = pred.clone()
                # scale_boxes(inputs[0][si].shape[2:], predn[:, :4], shape, meta["shapes"][si][1]) # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])
                    # scale_boxes(inputs[0][si].shape[2:], tbox, shape, meta["shapes"][si][1])
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)
                    if cfg.VAL.PLOT:
                        confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

                # Save/log
                if cfg.SAVE.SAVE_TXT:
                    save_one_txt(predn, cfg.SAVE.SAVE_CONF, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
                # 后面再实现 
                if cfg.SAVE.SAVE_JSON:
                    pass
        
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=cfg.VAL.PLOT, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        if len(cfg.DATA.FILTER_C) > 0:
            mp, mr, map50, map = p[cfg.DATA.FILTER_C].mean(), r[cfg.DATA.FILTER_C].mean(), ap50[cfg.DATA.FILTER_C].mean(), ap[cfg.DATA.FILTER_C].mean()
        else:
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=cfg.MODEL_PARA.CLASSES)  # number of targets per class
    else:
        nt = torch.zeros(1)


    # Print results
    # if du.is_master_proc():
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (cfg.MODEL_PARA.CLASSES < 50 and not training)) and cfg.MODEL_PARA.CLASSES  > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))


    # Plots
    if cfg.VAL.PLOT:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if cfg.SAVE.SAVE_TXT else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

    maps = np.zeros(cfg.MODEL_PARA.CLASSES) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(val_loader)).tolist()), maps

def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    
    # Directories
    save_dir = str(increment_path(Path(cfg.SAVE.PROJECT) / cfg.SAVE.NAME, exist_ok=cfg.SAVE.EXIST_OK))
    w = Path(save_dir) / 'weight'
    confusionMatrix = Path(save_dir) / 'confusion_matrix'
    w.mkdir(parents=True, exist_ok=True)  # make dir
    confusionMatrix.mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pyth', w / 'best.pyth'

    # Set up environment.
    du.init_distributed_training(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Print config
    if du.is_master_proc():
        LOGGER.info("Train with config:")
        LOGGER.info(pprint.pformat(cfg))
    # Build the video model and print model statistics.
    model = build_model(cfg)

    # flops, params = 0.0, 0.0
    # if du.is_master_proc() and cfg.LOG_MODEL_INFO:
    #     flops, params = misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer

    nl = float(model.model[-1].nl if hasattr(model, 'model') else model.module.model[-1].nl)
    imgsz = cfg.DATA.TRAIN_CROP_SIZE

    cfg.LOSS.BOX *= 3. / nl
    cfg.LOSS.CLS *= cfg.MODEL_PARA.CLASSES / 80. * 3. / nl
    cfg.LOSS.OBJ *= (imgsz / 256) ** 2 * 3. / nl 

    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION) # 如果出错，就来看这里

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(w):
        LOGGER.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(w)
        if last_checkpoint is not None:
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = 0
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader, dataset = loader.construct_loader(cfg, "train")
    val_loader, _ = loader.construct_loader(cfg, "val")

    # check_anchors(dataset, model=model, thr=cfg.LOSS.ANCHOR_T, imgsz=cfg.DATA.TRAIN_CROP_SIZE)

    # Perform the training loop.
    LOGGER.info("Start epoch: {}".format(start_epoch + 1))

    # define loss
    compute_loss = ComputeLoss(cfg, model)

    if du.is_master_proc():
        best_fitness = 0.0

    # epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        if hasattr(train_loader.dataset, "_set_epoch_num"):
            train_loader.dataset._set_epoch_num(cur_epoch)
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            cur_epoch,
            cfg,
            cfg.SOLVER.MAX_EPOCH,
            compute_loss,
        )
        results, maps = eval_epoch(val_loader, model, cur_epoch, cfg, train_loader, compute_loss)
        if du.is_master_proc():
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1)) # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            # stop  TO DO..
            if fi > best_fitness:
                best_fitness = fi

        # Save model
        # Save last, best and delete
        cu.save_checkpoint(last, model, optimizer, cur_epoch, cfg, scaler if cfg.TRAIN.MIXED_PRECISION else None)
        if du.is_master_proc():
            if best_fitness == fi:
                cu.save_checkpoint(best, model, optimizer, cur_epoch, cfg, scaler if cfg.TRAIN.MIXED_PRECISION else None)
