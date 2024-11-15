import pkg_resources as pkg
import os
from utils.general import colorstr
from torch.utils.tensorboard import SummaryWriter

LOGGERS = ('csv', 'tb', 'wandb', 'clearml')
RANK = int(os.getenv('RANK', -1))

