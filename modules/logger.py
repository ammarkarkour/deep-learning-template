import sys
sys.path.append('../')

import torch
from loggers.logger import Loggers
from config.logger import LogConfig

class LogModules:
    Writer = Loggers.TensorboardWriter
    Log_step = 100
    Checkpoint = torch.load(LogConfig.PRETRAINED_WEIGHTS, map_location='cpu') \
                 if LogConfig.PRETRAINED_WEIGHTS else {}