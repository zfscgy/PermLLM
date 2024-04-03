from typing import Union, Tuple, Collection
import numpy as np
import torch
from torch import nn


def copy_param(param_from: nn.Parameter, param_to: nn.Parameter):
    param_to.data = param_from.data.detach().clone()


if __name__ == "__main__":
    pass