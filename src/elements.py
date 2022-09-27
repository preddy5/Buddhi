import argparse
import os
import random
import shutil
import time
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random

import numpy as np

class bddi_if(nn.Module):
    def __init__(self, layer, threshold=1):
        super(bddi_if, self).__init__()
        self.layer = layer
        self.threshold = threshold
        
    def activate(self, c_old, c_pred):
        c = c_old + c_pred
        # Note this is not bddi activation.
        # This is an alternative with same forward pass output.
        c_spike = torch.zeros_like(c)
        c_spike[c>=self.threshold] = self.threshold
        c_reminder = c - c_spike
        return c_spike, c_reminder

    def forward(self, inputs, state, proj_fn=None, return_pred=False):
        if type(proj_fn) != type(None):
            state_pred = proj_fn(self.layer(inputs))
        else:
            state_pred = self.layer(inputs).clamp(min=0, max=1.0)
        output, new_state = self.activate(state, state_pred)
        if return_pred:
            return output, new_state, state_pred
        return output, new_state


def create_conv_if(infeatures, outfeatures, filter_size, stride, padding=0, size=None, threshold=1):
    layer = nn.Conv2d(infeatures, outfeatures, filter_size, stride=stride, padding=padding)
    return bddi_if(layer, threshold=threshold)
