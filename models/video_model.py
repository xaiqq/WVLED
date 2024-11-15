import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import contextlib
import sys
from pathlib import Path

from .build import MODEL_REGISTRY
from .common import *
from utils.general import (make_divisible, LOGGER, check_version,
                           time_sync)
from utils.autoanchor import check_anchor_order
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill
from utils.general import read_pbtxt
import utils.distributed as du

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

class Detect(nn.Module):
    stride = None
    dynamic = False
    export = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True): # detection layer
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.nl)] # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)] # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2)) # (nl, na, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch) # output conv
        self.inplace = inplace
    
    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape # x(bs, 255, 14, 14)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0,1,3,4,2).contiguous()

            if not self.training: # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)

                z.append(y.view(bs, self.na * nx * ny, self.no))
        return x if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)
                


    def _make_grid(self, nx=14, ny=14, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
                   

class BaseModule(nn.Module):
    # WVLED base module
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize) # single-scale inference, tarin
    
    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            # if profile:
            #     self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            # visualize
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x, ), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
    
    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

@MODEL_REGISTRY.register()
class WVLED(BaseModule):
    # WVLED model
    def __init__(self, cfg):
        super().__init__()
        ch = cfg.MODEL.CH
        self.model, self.save = self._construct_model(cfg, [ch])
        self.inplace = cfg.MODEL.INPLACE
        self.names = read_pbtxt(Path(cfg.DATA.ANNOTATION_DIR) / Path(cfg.DATA.LABEL_MAP_FILE))

        # Build strides, anchors
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 224
            t = cfg.DATA.NUM_FRAMES
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, t, s, s))])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.MODEL.ZERO_INIT_FINAL_BN,
            cfg.MODEL.ZERO_INIT_FINAL_CONV,
        )
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (224 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _construct_model(self, cfg, ch):
        anchors = cfg.ANCHORS
        nc = cfg.MODEL_PARA.CLASSES
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
        no = na * (nc + 5)

        layers, save, c2 = [], [], ch[-1] # layers, savelist, ch out

        for i, (f, n, m, args) in enumerate(cfg.MODEL.BACKBONE + cfg.MODEL.HEAD):
            m = eval(m) if isinstance(m, str) else m
            for j, a in enumerate(args):
                with contextlib.suppress(NameError):
                    args[j] = eval(a) if isinstance(a, str) else a

            if m in {BasicBlock, BasicBlock_2 ,Conv_3d}:
                c1, c2, k= ch[f], args[0], args[1]
                if c2 != no:
                    make_divisible(c2, 8)
                if k is None:
                    T = args[2]
                    if m is Conv_3d:
                        k = (T, 3, 3)
                        p = (0, k[1] // 2, k[2] // 2)
                        s = 1
                        args = [c1, c2, k, s, p, *args[3:]]
                    else:
                        k = T
                        args = [c1, c2, k, *args[3:]]
                else:
                    args = [c1, c2, *args[1:]]
            elif m in [Conv_2d, SpatialBlock, SpatialBlock2, TemporalBlock, TemporalBlock2, TemporalBlock_, FeatureBlock, To_Head1, SPPF,FusionLayer]:
                c1, c2 = ch[f], args[0]
                args = [c1, c2, *args[1:]]
            elif m is C3:
                c1, c2 = ch[f], args[0]
                if c2 != no:
                    c2 = make_divisible(c2, 8)
                args = [c1, c2, *args[1:]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m is Add:
                c2 = ch[f[0]]
            elif m in {Detect}:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):
                    args[1] = [list(range(args[1] * 2))] * len(f)
            elif m in {FusionT, FusionS}:
                c1, c2 = ch[f[0]], args[0]
                args = [c1, c2, *args[1:]]
            else:
                c2 = ch[f]
            
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            np = sum(x.numel() for x in m_.parameters())  # number params
            m_.i, m_.f, m_.type, m_np = i, f, t, np
            if du.is_master_proc():
                LOGGER.info(f'{i:>3}{str(f):>18}{n:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)



def init_weights(
    model, fc_init_std=0.01, zero_init_final_bn=True, zero_init_final_conv=False
):
    """
    Performs ResNet style weight initialization.
    Args:
        fc_init_std (float): the expected standard deviation for fc layer.
        zero_init_final_bn (bool): if True, zero initialize the final bn for
            every bottleneck.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            # Note that there is no bias due to BN
            if hasattr(m, "final_conv") and zero_init_final_conv:
                m.weight.data.zero_()
            else:
                """
                Follow the initialization method proposed in:
                {He, Kaiming, et al.
                "Delving deep into rectifiers: Surpassing human-level
                performance on imagenet classification."
                arXiv preprint arXiv:1502.01852 (2015)}
                """
                c2_msra_fill(m)

        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            if (
                hasattr(m, "transform_final_bn")
                and m.transform_final_bn
                and zero_init_final_bn
            ):
                batchnorm_weight = 0.0
            else:
                batchnorm_weight = 1.0
            if m.weight is not None:
                m.weight.data.fill_(batchnorm_weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            m.inplace = True
        if isinstance(m, nn.Linear):
            if hasattr(m, "xavier_init") and m.xavier_init:
                c2_xavier_fill(m)
            else:
                m.weight.data.normal_(mean=0.0, std=fc_init_std)
            if m.bias is not None:
                m.bias.data.zero_()

Model = WVLED