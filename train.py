import torch
import torch.nn as nn
from pathlib import Path
import sys
import os
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
     sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.general import (print_args)
from utils.checkpoint import load_config
from utils.misc import launch_job
from configs.defaults import assert_and_infer_cfg
from tools.train_net import train
from tools.train_net_ava import train as train_ava

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_files", help="Path to the config files", default=["configs/UCF101_24_base.yaml"], nargs="+")
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    if RANK in {-1, 0}:
        print_args()
    for path_to_config in opt.cfg_files:
        cfg = load_config(opt, path_to_config)
        cfg = assert_and_infer_cfg(cfg)
    # Training
    if cfg.TRAIN.ENABLE:
        if cfg.TRAIN.DATASET == "ava":
            launch_job(cfg=cfg, init_method=opt.init_method, func=train_ava)
        else:
            launch_job(cfg=cfg, init_method=opt.init_method, func=train)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

