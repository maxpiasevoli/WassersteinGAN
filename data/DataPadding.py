import torch
import pandas as pd
import numpy as np

def _isEven(num):

    return num % 2 == 0

# return next multiple of 16 that is larger than num
def _nextMult16(num):

    return (num // 16 + 1) * 16

# given the current dimension return the needed padding on each side
# for this dimension
def _determineDimPadding(dim):

    is_even = _isEven(dim)
    next_mult = _nextMult16(dim)

    if is_even:

        padding_lt = (next_mult - dim) / 2
        padding_rb = padding_lt

    else:

        padding_lt = (next_mult - dim) // 2
        padding_rb = padding_lt + 1

    return padding_lt, padding_rb

# dataV is a tensor variable that is to be padded so that it's dimensions
# are a mutliple of 16. This is mainly used for the DCGAN
def padData(dataV):

    row_dim = dataV.shape[1]
    col_dim = dataV.shape[2]
    padding_left, padding_right = _determineDimPadding(row_dim)
    padding_top, padding_bottom = _determineDimPadding(col_dim)
    torch_padder = torch.nn.ZeroPad2d((padding_left, padding_right,
                                       padding_top, padding_bottom))
    return torch_padder(dataV)
