import math
import torch.nn as nn
import sys
import os
from pathlib import Path
from fvcore.common.config import CfgNode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #

_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Kill training if loss explodes over this ratio from the previous 5 measurements.
# Only enforced if > 0.0
_C.TRAIN.KILL_LOSS_EXPLOSION_FACTOR = 0.0

# Dataset
_C.TRAIN.DATASET = "ava"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 8

_C.TRAIN.DEVICE = 0

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# If True, use FP16 for activations
_C.TRAIN.MIXED_PRECISION = False


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.VAL = CfgNode()

_C.VAL.COMPUTE_LOSS = True

_C.VAL.SAVE_HYBRID = False

_C.VAL.CONF_THRES = 0.001

_C.VAL.IOU_THRES = 0.6

_C.VAL.MAX_DET = 300

_C.VAL.SINGLE_CLS=False

_C.VAL.PLOT = True


# ---------------------------------------------------------------------------- #SOLVER.WARMUP_EPOCHS
# Misc options
# ---------------------------------------------------------------------------- #

_C.OUTPUT_DIR = "."

_C.RNG_SEED = 1

_C.NUM_GPUS = 8

_C.TASK = ""

# Log period in iters.
_C.LOG_PERIOD = 10

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# ---------------------------------------------------------------------------- #
# CONSTRACTION PARA OF MODEL
# ---------------------------------------------------------------------------- #
_C.MODEL_PARA = CfgNode()
_C.MODEL_PARA.MODEL_NAME = 'WVLED'

_C.MODEL_PARA.DIM_IN = 3

# T dimention
_C.MODEL_PARA.T = 8

# number of classes
_C.MODEL_PARA.CLASSES = 80

_C.MODEL_PARA.FROZEN_BN = False

# Loss function.
_C.MODEL_PARA.LOSS_FUNC = "cross_entropy"

# If True, detach the final fc layer from the network, by doing so, only the
# final fc layer will be trained.
_C.MODEL_PARA.DETACH_FINAL_FC = False

# If True, AllReduce gradients are compressed to fp16
_C.MODEL_PARA.FP16_ALLREDUCE = False

# ---------------------------------------------------------------------------- #
# Anchors
# ---------------------------------------------------------------------------- #
# _C.ANCHORS = [[20,23, 32,44, 38,68], [54,72, 47,96, 72,101]]
_C.ANCHORS = [[15, 16, 21, 35, 38, 25], [20,23, 32,44, 38,68], [54,72, 47,96, 72,101]]
# _C.ANCHORS = [[10,13, 16,30, 33,23], [20,23, 32,44, 38,68]]
# ---------------------------------------------------------------------------- #
# MODEL
# ---------------------------------------------------------------------------- #
_C.MODEL = CfgNode()

_C.MODEL.INPLACE = True

# Input channel
_C.MODEL.CH = 3

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

_C.MODEL.ZERO_INIT_FINAL_BN = False

_C.MODEL.ZERO_INIT_FINAL_CONV = False

_C.MODEL.BACKBONE = [
    # [from, number, module, args]
    [-1, 1, "SpatialBlock", [64, 3]], # [64, 32, 112, 112]
    [-1, 2, "TemporalBlock", [64, 3, 1, 1]],
    [[-1, 0], 1, "Add", []],

    [-1, 1, "SpatialBlock", [128, 3]], # [128, 32, 56, 56] 3
    [-1, 1, "TemporalBlock", [128]],
    [-1, 2, "TemporalBlock", [128, 3, 1, 1]],
    [[-1, 4], 1, "Add", []],

    [-1, 1, "SpatialBlock", [256, 3]], # [256, 16, 28, 28] 7
    [-1, 1, "TemporalBlock", [256]],
    [-1, 2, "TemporalBlock", [256, 3, 1, 1]],
    [[-1, 8], 1, "Add", []],

    [-1, 1, "SpatialBlock", [512, 3]], # 6 [512, 8, 14, 14] 11
    [-1, 1, "TemporalBlock", [512]], # [512, 4, 14, 14]
    [-1, 2, "TemporalBlock", [512, 3, 1, 1]],
    [[-1, 12], 1, "Add", []],

    [-1, 1, "SpatialBlock", [512, 3, 2, 1, 2]], # [512, 4, 7, 7]
    [-1, 1, "TemporalBlock", [512]], # [512, 2, 7, 7]
    [-1, 2, "TemporalBlock", [512, 3, 1, 1]],
    [[-1, 16], 1, "Add", []], # 18
]

_C.MODEL.HEAD = [
    # [from, number, module, args]
    [13, 1, "SpatialBlock2", [512]],
    [-1, 1, "TemporalBlock_", [1024, 4]],
    [18, 1, "TemporalBlock_", [1024, 2]],
    [[-1, 20], 1, "Concat", [1]], # 22
    [21, 1, "nn.Upsample", [None, 2, 'nearest']],

    [9, 1, "SpatialBlock2", [256]],
    [-1, 1, "TemporalBlock_", [512, 8]],
    [13, 1, "TemporalBlock_", [512, 4]],
    [[-1, 23, 25], 1, "Concat", [1]], # 27
    [26, 1, "nn.Upsample", [None, 2, 'nearest']],

    [5, 1, "SpatialBlock2", [128]],
    [-1, 1, "TemporalBlock_", [256, 16]],
    [9, 1, "TemporalBlock_", [256, 8]],
    [[-1, 28, 30], 1, "Concat", [1]], # 32

    [[22, 27, 32], 1, "Detect", [_C.MODEL_PARA.CLASSES, _C.ANCHORS]],
]

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #

_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.05

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = False

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = False

# Clip gradient at this value before optimizer update
_C.SOLVER.CLIP_GRAD_VAL = None

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = None

# LARS optimizer
_C.SOLVER.LARS_ON = False

# The layer-wise decay of learning rate. Set to 1. to disable.
_C.SOLVER.LAYER_DECAY = 1.0

# Adam's beta
_C.SOLVER.BETAS = (0.9, 0.999)

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized. `NUM_SYNC_DEVICES` cannot be larger than number of
# devices per machine; if global sync is desired, set `GLOBAL_SYNC`.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.NUM_SYNC_DEVICES = 1

# Parameter for NaiveSyncBatchNorm. Setting `GLOBAL_SYNC` to True synchronizes
# stats across all devices, across all machines; in this case, `NUM_SYNC_DEVICES`
# must be set to None.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.GLOBAL_SYNC = False


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# The separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR = " "

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# Period of frame
_C.DATA.KEY_FRAME = 15

_C.DATA.MEAN = [0.45, 0.45, 0.45]

_C.DATA.STD = [0.225, 0.225, 0.225]

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# how many samples (=clips) to decode from a single video
_C.DATA.TRAIN_CROP_NUM_TEMPORAL = 1

# how many spatial samples to crop from a single clip
_C.DATA.TRAIN_CROP_NUM_SPATIAL = 1

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

# loader can read .csv file in chunks of this chunk size
_C.DATA.LOADER_CHUNK_SIZE = 0

# if LOADER_CHUNK_SIZE > 0, define overall length of .csv file
_C.DATA.LOADER_CHUNK_OVERALL_SIZE = 0

# for chunked reading, dataloader can skip rows in (large)
# training csv file
_C.DATA.SKIP_ROWS = 0

_C.DATA.FILTER_C = []


_C.DATA.ANNOTATION_DIR = str(ROOT) + "/data"
_C.DATA.LABEL_MAP_FILE = "ava/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt"

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
_C.SAVE = CfgNode()

# save to project/name
_C.SAVE.NAME = 'exp'

# Save Dir
_C.SAVE.SAVE_DIR = ''

# Save results to *.txt
_C.SAVE.SAVE_TXT = False

# Save a COCO-JSON results file
_C.SAVE.SAVE_JSON = False

_C.SAVE.PROJECT = str(ROOT / 'runs/val')

# existing project/name ok, do not increment
_C.SAVE.EXIST_OK = True

_C.SAVE.SAVE_CONF = False

# -----------------------------------------------------------------------------
# AVA Dataset options
# -----------------------------------------------------------------------------
_C.AVA = CfgNode()

# If use BGR as the format of input frames.
_C.AVA.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.AVA.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to do horizontal flipping during test.
_C.AVA.TEST_FORCE_FLIP = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.AVA.TRAIN_PCA_JITTER_ONLY = True

# This option controls the score threshold for the predicted boxes to use.
_C.AVA.DETECTION_SCORE_THRESH = 0.9

# Directory path of frames.
_C.AVA.FRAME_DIR = str(ROOT) + "/data/ava/frames"
_C.AVA.ANNOTATION_DIR = (
    str(ROOT) + "/data/ava/annotations"
)

# Directory path for files of frame lists.
_C.AVA.FRAME_LIST_DIR = (
    str(ROOT) + "/data/ava/frame_lists/"
)

# Filenames of training samples list files.
_C.AVA.TRAIN_LISTS = ["train.csv"]

# Filenames of test samples list files.
_C.AVA.TEST_LISTS = ["val.csv"]

# Filenames of box list files for training. Note that we assume files which
# contains predicted boxes will have a suffix "predicted_boxes" in the
# filename.
_C.AVA.TRAIN_GT_BOX_LISTS = ["ava_train_v2.2.csv"]
_C.AVA.TRAIN_PREDICT_BOX_LISTS = []

_C.AVA.VAL_GT_BOX_LISTS = ["ava_val_v2.2.csv"]

# Filenames of box list files for test.
_C.AVA.TEST_PREDICT_BOX_LISTS = []

_C.AVA.FULL_TEST_ON_VAL = False

# Backend to process image, includes `pytorch` and `cv2`.
_C.AVA.IMG_PROC_BACKEND = "pytorch"

# The name of the file to the ava exclusion.
_C.AVA.EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv"

# The name of the file to the ava label map.
_C.AVA.LABEL_MAP_FILE = "ava_action_list_v2.2_for_activitynet_2019.pbtxt"

# The name of the file to the ava groundtruth.
_C.AVA.GROUNDTRUTH_FILE = "ava_val_v2.2.csv"


# -----------------------------------------------------------------------------
# HMDB Dataset options
# -----------------------------------------------------------------------------
_C.HMDB = CfgNode()

# If use BGR as the format of input frames.
_C.HMDB.BGR = False

_C.HMDB.TRAIN_USE_COLOR_AUGMENTATION = False

_C.HMDB.TRAIN_PCA_JITTER_ONLY = True

# Whether to do horizontal flipping during test.
_C.HMDB.TEST_FORCE_FLIP = False

# Directory path for files of frame lists.
_C.HMDB.FRAME_DIR = (
    str(ROOT) + "/data/HMDB_mini/frames/"
)

_C.HMDB.ANNOTATIONS_DIR = (
    str(ROOT) + "/data/HMDB_mini/annotations/"
)

_C.HMDB.IMG_PROC_BACKEND = "pytorch"

_C.HMDB.LABEL_MAP_FILE = "hmdb_mini_action_list.pbtxt"



# -----------------------------------------------------------------------------
# JHMDB Dataset options
# -----------------------------------------------------------------------------
_C.JHMDB = CfgNode()

# If use BGR as the format of input frames.
_C.JHMDB.BGR = False

_C.JHMDB.TRAIN_USE_COLOR_AUGMENTATION = False

_C.JHMDB.TRAIN_PCA_JITTER_ONLY = True

# Whether to do horizontal flipping during test.
_C.JHMDB.TEST_FORCE_FLIP = False

# Directory path for files of frame lists.
_C.JHMDB.FRAME_DIR = (
    str(ROOT) + "/data/JHMDB/Frames/"
)

_C.JHMDB.ANNOTATIONS_FILE = (
    str(ROOT) + "/data/JHMDB/JHMDB-GT.pkl"
)

_C.JHMDB.ANNOTATIONS_DIR = (
    str(ROOT) + "/data/JHMDB/annotations"
)

_C.JHMDB.IMG_PROC_BACKEND = "pytorch"

_C.JHMDB.LABEL_MAP_FILE = "jhmdb_action_list.pbtxt"


# -----------------------------------------------------------------------------
# UCF101_24 Dataset options
# -----------------------------------------------------------------------------
_C.UCF101_24 = CfgNode()

# If use BGR as the format of input frames.
_C.UCF101_24.BGR = False

_C.UCF101_24.TRAIN_USE_COLOR_AUGMENTATION = True

# Whether to do horizontal flipping during test.
_C.UCF101_24.TEST_FORCE_FLIP = False

# Directory path for files of frame lists.
_C.UCF101_24.FRAME_DIR = (
    str(ROOT) + "/data/UCF101_24/Frames/"
)

_C.UCF101_24.ANNOTATIONS_FILE = (
    str(ROOT) + "/data/UCF101_24/UCF101v2-GT.pkl"
)

_C.UCF101_24.ANNOTATIONS_DIR = (
    str(ROOT) + "/data/UCF101_24/annotations"
)

_C.UCF101_24.IMG_PROC_BACKEND = "pytorch"

_C.UCF101_24.LABEL_MAP_FILE = "ucf101_24_action_list.pbtxt"

# ---------------------------------------------------------------------------- #
# Multigrid training options
# See https://arxiv.org/abs/1912.00998 for details about multigrid training.
# ---------------------------------------------------------------------------- #
_C.MULTIGRID = CfgNode()

# Multigrid training allows us to train for more epochs with fewer iterations.
# This hyperparameter specifies how many times more epochs to train.
# The default setting in paper trains for 1.5x more epochs than baseline.
_C.MULTIGRID.EPOCH_FACTOR = 1.5

# Enable short cycles.
_C.MULTIGRID.SHORT_CYCLE = False

# Short cycle additional spatial dimensions relative to the default crop size.
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5**0.5]

_C.MULTIGRID.LONG_CYCLE = False

# (Temporal, Spatial) dimensions relative to the default shape.
_C.MULTIGRID.LONG_CYCLE_FACTORS = [
    (0.25, 0.5**0.5),
    (0.5, 0.5**0.5),
    (0.5, 1),
    (1, 1),
]

# While a standard BN computes stats across all examples in a GPU,
# for multigrid training we fix the number of clips to compute BN stats on.
# See https://arxiv.org/abs/1912.00998 for details.
_C.MULTIGRID.BN_BASE_SIZE = 8

# Multigrid training epochs are not proportional to actual training time or
# computations, so _C.TRAIN.EVAL_PERIOD leads to too frequent or rare
# evaluation. We use a multigrid-specific rule to determine when to evaluate:
# This hyperparameter defines how many times to evaluate a model per long
# cycle shape.
_C.MULTIGRID.EVAL_FREQ = 3

# No need to specify; Set automatically and used as global variables.
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0
_C.MULTIGRID.DEFAULT_B = 0
_C.MULTIGRID.DEFAULT_T = 0
_C.MULTIGRID.DEFAULT_S = 0

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False

# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #
_C.DETECTION = CfgNode()

# Whether enable video detection.
_C.DETECTION.ENABLE = True

# Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
_C.DETECTION.ALIGNED = True

# Spatial scale factor.
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

# RoI tranformation resolution.
_C.DETECTION.ROI_XFORM_RESOLUTION = 7


# ---------------------------------------------------------------------------- #
# Augmentation options.
# ---------------------------------------------------------------------------- #
_C.AUG = CfgNode()

# Number of repeated augmentations to used during training.
# If this is greater than 1, then the actual batch size is
# TRAIN.BATCH_SIZE * AUG.NUM_SAMPLE.
_C.AUG.NUM_SAMPLE = 1

# Not used if using randaug.
_C.AUG.COLOR_JITTER = 0.4

# RandAug parameters.
_C.AUG.AA_TYPE = "rand-m9-mstd0.5-inc1"

# Interpolation method.
_C.AUG.INTERPOLATION = "bicubic"

# Probability of random erasing.
_C.AUG.RE_PROB = 0.25

# Random erasing mode.
_C.AUG.RE_MODE = "pixel"

# Random erase count.
_C.AUG.RE_COUNT = 1

# Do not random erase first (clean) augmentation split.
_C.AUG.RE_SPLIT = False

# Whether to generate input mask during image processing.
_C.AUG.GEN_MASK_LOADER = False

# If True, masking mode is "tube". Default is "cube".
_C.AUG.MASK_TUBE = False

# If True, masking mode is "frame". Default is "cube".
_C.AUG.MASK_FRAMES = False

# The size of generated masks.
_C.AUG.MASK_WINDOW_SIZE = [8, 7, 7]

# The ratio of masked tokens out of all tokens. Also applies to MViT supervised training
_C.AUG.MASK_RATIO = 0.0

# The maximum number of a masked block. None means no maximum limit. (Used only in image MaskFeat.)
_C.AUG.MAX_MASK_PATCHES_PER_BLOCK = None

# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = False


# ---------------------------------------------------------------------------- #
# MipUp options.
# ---------------------------------------------------------------------------- #
_C.MIXUP = CfgNode()

# Whether to use mixup.
_C.MIXUP.ENABLE = False

# Mixup alpha.
_C.MIXUP.ALPHA = 0.8

# Cutmix alpha.
_C.MIXUP.CUTMIX_ALPHA = 1.0

# Probability of performing mixup or cutmix when either/both is enabled.
_C.MIXUP.PROB = 1.0

# Probability of switching to cutmix when both mixup and cutmix enabled.
_C.MIXUP.SWITCH_PROB = 0.5

# Label smoothing.
_C.MIXUP.LABEL_SMOOTH_VALUE = 0.1

# ---------------------------------------------------------------------------- #
# Compute Losses.
# ---------------------------------------------------------------------------- #
_C.LOSS = CfgNode()

_C.LOSS.AUTOBALANCE=False

# cls BCELoss positive_weight
_C.LOSS.CLS_PW=1.0

# obj BCELoss positive_weight
_C.LOSS.OBJ_PW=1.0

_C.LOSS.LABEL_SMOOTHING=0.0

# focal loss gamma
_C.LOSS.FL_GAMMA = 1.5

_C.LOSS.ANCHOR_T = 4.0

# box loss gain
_C.LOSS.BOX = 0.05

# cls loss gain
_C.LOSS.CLS = 0.3

# obj loss gain (scale with pixels)
_C.LOSS.OBJ = 0.7



def assert_and_infer_cfg(cfg):
    
    return cfg

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()