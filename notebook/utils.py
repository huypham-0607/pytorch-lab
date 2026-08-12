import random

import numpy as np
import torch

def seed_everything(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
