import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.general import yaml_load, read_pbtxt, check_suffix


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv_2d(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Conv_3d(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.Conv = nn.Conv3d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.Conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
                               x = self.drop(x)
        return x
# -----------------------------------------------------------normal


class BasicBlock(nn.Module):
    def __init__(self, c1, c2, k=1, t=8, p=None):
        super().__init__()
        c_ = c2 // 2
        self.conv1 = Conv_3d(c1, c_, k=1)
        self.conv2 = Conv_3d(c_, c2, k=1)
        self.m = nn.ModuleList()
        for _ in range(t):
            self.m.append(Conv_3d(c_, c_, k=(k, 1, 1), s=(1, 1, 1), p=(0, 1, 1)))
    def forward(self, x):
        x = self.conv1(x)
        x = [mi(x) for mi in self.m]
        return self.conv2(torch.cat(x, dim=2))

# class BasicBlock(nn.Module):
#     def __init__(self, c1, c2, k=1, p=None):
#         super().__init__()
#         self.conv1 = Conv_3d(c1, c2, k=(3, 1, 1), s=(1, 1, 1), p=(1, 0, 0))
#     def forward(self, x):
#         y = torch.sum(self.conv1(x) * x, dim=2)[:, :, None, :, :]
#         return torch.cat([y, x], dim=2)


class BasicBlock_2(nn.Module):
    def __init__(self, c1, c2, k=1, p=None, g=1, act=True):
        super().__init__()
        c_ = c2 * 2
        self.conv1 = Conv_3d(c1, c_, k=1)
        self.conv2 = Conv_3d(c_, c2, k=1)
        self.res = Conv_3d(c1, c2, k=(k, 1, 1), s=(1, 1, 1), p=(0, 0, 0))
    def forward(self, x):
        # return torch.cat([x + self.conv2(self.conv1(x)), self.res(x)], dim=2)
        return x + self.conv2(self.conv1(x)) * self.res(x)

# class FeatureBlock_(nn.Module):
#     def __init__(self, c1, c2):
#         super().__init__()
#         c_ = c2 * 2
#         self.conv1 = Conv_3d(c1, c_, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
#         self.conv2 = Conv_3d(c_, c2, k=1)
#     def forward(self, x):
#         return x + self.conv2(self.conv1(x))

# class FeatureBlock(nn.Module):
#     def __init__(self, c1, c2, n):
#         super().__init__()
#         self.m = nn.Sequential(*(FeatureBlock_(c1, c2) for _ in range(n)))
#     def forward(self, x):
#         return x + self.m(x)


class Bottleneck_3D(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e) # hidden channels
        self.cv1 = Conv_3d(c1, c_, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
        self.cv2 = Conv_3d(c_, c1, k=1)
        self.cv3 = Conv_3d(c1, c2, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
        # self.cv2 = Conv_3d(c_, c2, k=3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(x + self.cv2(self.cv1(x))) if self.add else self.cv3(x + self.cv2(self.cv1(x)))

# class Bottleneck_WH(nn.Module):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e) # hidden channels
#         self.cv1 = Conv_3d(c1, c_, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
#         self.cv2 = Conv_3d(c_, c2, k=1)
#         # self.cv2 = Conv_3d(c_, c2, k=3)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# class Bottleneck_T(nn.Module):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e) # hidden channels
#         self.cv1 = Conv_3d(c1, c_, k=(3, 1, 1), s=(1, 1, 1), p=(1, 0, 0))
#         self.cv2 = Conv_3d(c_, c2, k=1)
#         # self.cv2 = Conv_3d(c_, c2, k=3)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class FeatureBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = Conv_3d(c1, c2, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
    def forward(self, x):
        return x + self.conv1(x)


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for WVLED by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv_3d(c1, c_, 1, 1)
        self.cv2 = Conv_3d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool3d(kernel_size=(3, k, k), stride=(1, 1, 1), padding=(1, k // 2, k // 2))

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class SpatialBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=1, n=1):
        super().__init__()
        self.conv1 = Conv_3d(c1, c2, k=(1, k, k), s=(1, s, s), p=(0, p, p))
        self.m = nn.Sequential(*(Bottleneck_3D(c2, c2, e=1.0) for _ in range(n)))
    def forward(self, x):
        return self.m(self.conv1(x))

class SpatialBlock2(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=1):
        super().__init__()
        self.conv1 = Conv_3d(c1, c2, k=(1, k, k), s=(1, s, s), p=(0, p, p))
    def forward(self, x):
        return self.conv1(x)


# class TemporalBlock_(nn.Module):
#     def __init__(self, c1, c2, t, k=1):
#         super().__init__()
#         self.conv1 = Conv_2d(c1 * t, c2, k=k)
#     def forward(self, x):
#         x_b, x_c, x_t, x_w, x_h = x.shape
#         x = x.reshape(x_b, x_c * x_t, x_w, x_h)
#         return self.conv1(x)

class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)


class TemporalBlock_(nn.Module):
    def __init__(self, c1, c2, t=8):
        super().__init__()
        self.conv1 = Conv_3d(c1, c1, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
        self.conv2 = Conv_2d(c1 * t, c2, 1, 1)
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x_b, x_c, x_t, x_w, x_h = x.shape
        x = x.view(x_b, x_c * x_t, x_w, x_h)
        return self.conv2(x)

class Add(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x[0] + x[1]

class TemporalBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=1):
        super().__init__()
        self.conv1 = Conv_3d(c1, c2, k=(k, 3, 3), s=(s, 1, 1), p=(p, 1, 1))
    def forward(self, x):
        return self.conv1(x)

class TemporalBlock2(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = Conv_3d(c1, c2, k=3, s=1, p=1)
    def forward(self, x):
        return x + self.conv1(x)

class FusionLayer(nn.Module):
    def __init__(self, c1, c2, t):
        super().__init__()
        self.conv1 = Conv_3d(c1, c2, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
        self.conv2 = Conv_3d(c2, c2, k=(t, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
    def forward(self, x):
        return self.conv2(self.conv1(x)).squeeze(2)


# class FusionT(nn.Module):
#     def __init__(self, c1, c2, e=0.5):
#         super().__init__()
#         self.conv1 = Conv_3d(c1, c2, k=(1, 3, 3), s=(1, 2, 2), p=(0, 1, 1))
#         self.conv2 = Conv_2d(c2 * 12, c2, k=1)
#     def forward(self, x):
#         x, y, z = x[0], x[1], x[2]
#         x = self.conv1(x)
#         x_b, x_c, x_t, x_w, x_h = x.shape
#         x = x.reshape(x_b, x_c * x_t, x_w, x_h)
#         return self.conv2(torch.cat([x, y, z], dim=1))

class FusionT(nn.Module):
    def __init__(self, c1, c2, c_, e=0.5):
        super().__init__()
        self.conv1 = Conv_3d(c1, c_, k=(1, 3, 3), s=(1, 2, 2), p=(0, 1, 1))
    def forward(self, x):
        x, y, z = x[0], x[1], x[2]
        x = self.conv1(x)
        x_b, x_c, x_t, x_w, x_h = x.shape
        x = x.view(x_b, x_c * x_t, x_w, x_h)
        return torch.cat([x, y, z], dim=1)

class FusionS(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
    def forward(self, x):
        return x[0] + x[1]


class To2D(nn.Module):
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.conv1 = C3(c1, c2, n=n)
    def forward(self, x):
        x = x[:, :, 0, :, :]
        return self.conv1(x)

class To_Head1(nn.Module):
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.conv1 = C3(c1 * n, c2, n=n)
    def forward(self, x):
        Tem_Dim = x.shape[2]
        y = []
        for t in range(Tem_Dim):
            y.append(x[:, :, t, :, :])
        return self.conv1(torch.cat(y, dim=1))

class ConvRes(nn.Module):
    def __init__(self, c1, c2, k=1):
        super().__init__()
        self.conv = Conv_3d(c1 * 2, c2, k=k)
    def forward(self, x):
        return self.conv(torch.cat([x, x], dim=1))

# # -----------------------------------------------------------压缩参数
# class BasicBlock(nn.Module):
#     def __init__(self, c1, c2, k=1, p=None, g=1, act=True):
#         super().__init__()
#         self.conv = Conv_3d(c1, c2, k=3, s=(1,2,2))
#         self.res1 = Conv_3d(c1, c2, k=(k, 5, 5), s=(1,2,2), p=(0, 2, 2))
#         self.res2 = Conv_3d(c1, c2, k=(k, 3, 3), s=(1,2,2), p=(0, 1, 1))
#     def forward(self, x):
#         return torch.cat([self.conv(x), self.res1(x), self.res2(x)], dim=2)
# -----------------------------------------------------------压缩T
# class BasicBlock(nn.Module):
#     def __init__(self, c1, c2, k=1, p=None, g=1, act=True):
#         super().__init__()
#         c_ = c2 // 2
#         self.conv1 = Conv_3d(c1, c_, k=3)
#         self.conv2 = Conv_3d(c_, c2, k=3, s=2)
#         self.res = Conv_3d(c1, c2, k=(k, 5, 5), s=(1,2,2), p=(0, 2, 2))
#     def forward(self, x):
#         return torch.cat([self.conv2(self.conv1(x)), self.res(x)], dim=2)

# class BasicBlock_(nn.Module):
#     def __init__(self, c1, c2, k=1, p=None, g=1, act=True):
#         super().__init__()
#         self.conv1 = Conv_3d(c1, c2, k=3)
#         self.res = Conv_3d(c1, c2, k=(k, 3, 3), s=(1,1,1), p=(0, 1, 1))
#     def forward(self, x):
#         return torch.cat([self.conv1(x), self.res(x)], dim=2)
# -----------------------------------------------------------
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e) # hidden channels
        self.cv1 = Conv_2d(c1, c_, 1, 1)
        self.cv2 = Conv_2d(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5): # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv_2d(c1, c_, 1, 1)
        self.cv2 = Conv_2d(c1, c_, 1, 1)
        self.cv3 = Conv_2d(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        if len(x.shape)==5 and x.shape[2] == 1:
            x = x.squeeze(2)
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3_Basic(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5): # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv_2d(c1, c_, 1, 1)
        self.cv2 = Conv_2d(c1, c_, 1, 1)
        self.cv3 = Conv_2d(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        if len(x.shape)==5 and x.shape[2] == 1:
            x = x.squeeze(2)
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)).unsqueeze(2)
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x):
        return torch.cat(x, self.d)
    


class DetectMultiBackend(nn.Module):
    # WVLED MultiBackend class for python inference on various backends
    def __init__(self, weights='WVLED.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        
        from models.experimental import attempt_load  # scoped to avoid circular import
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        else:
            # TO DO
            pass
        # class names
        if 'names' not in locals():
            names = read_pbtxt(data)

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, ims, augment=False, visualize=False):
        # WVLED MultiBackend inference
        b, ch, t, h, w = ims.shape  # batch, channel, height, width
        if self.fp16 and ims.dtype != torch.float16:
            ims = ims.half()  # to FP16
        if self.nhwc:
            ims = ims.permute(0, 2, 3, 4, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(ims)

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 8, 256, 256)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.general import is_url
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ['http', 'grpc']), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None