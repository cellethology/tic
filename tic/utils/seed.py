import random
import numpy as np
# try to import torch
try:
    import torch
except ImportError:
    pass

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if 'torch' in globals():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False