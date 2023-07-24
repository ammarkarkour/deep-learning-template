import sys
sys.path.append('../')

from optimization.optimizers import Optimizers
from optimization.criterions import Criterions

class OptimModules:
    Criterion = Criterions.CrossEntropy
    Optimizer = Optimizers.SGD