# Pytorch
import torch
import torch.nn as nn
# Local
import sys
# sys.path.append("DiffJPEG")
from modules import compress_jpeg, decompress_jpeg
from utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, height, width, subsampling=True):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        self.compress = compress_jpeg(subsampling=subsampling)
        self.decompress = decompress_jpeg(height, width, subsampling=subsampling)

    def forward(self, x, quality, differentiable):
        rounding = None
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        
        factor = quality_to_factor(quality)
        y, cb, cr = self.compress(x, factor, rounding)
        recovered = self.decompress(y, cb, cr, factor, rounding)
        return recovered