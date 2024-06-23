# utils.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# from model.digitization.loss import dice_loss
from loss import dice_loss as calc_loss

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array
