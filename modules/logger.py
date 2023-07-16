import sys
sys.path.append('../')

import torch
from loggers.logger import Loggers
from config.logger import LoggerConfig

class LogModules:
    Writer = Loggers.TensorboardWriter
    Log_step = 100
    Checkpoint = torch.load(LoggerConfig.PRETRAINED_WEIGHTS, map_location='cpu') \
                 if LoggerConfig.PRETRAINED_WEIGHTS else {}