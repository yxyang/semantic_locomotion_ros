"""Utilities for constructing neural networks."""

import numpy as np
import torch
import torch.nn as nn


def swish(x):
  return x * torch.sigmoid(x)


def get_affine_params(dim_in, dim_out):
  w = np.random.normal(loc=0,
                       scale=1.0 / (2.0 * np.sqrt(dim_in)),
                       size=(dim_in, dim_out)).astype(np.float32)
  w = nn.Parameter(torch.from_numpy(w), requires_grad=True)
  b = nn.Parameter(torch.zeros(dim_out, dtype=torch.float32),
                   requires_grad=True)
  return w, b
