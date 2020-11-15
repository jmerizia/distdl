import numpy as np
import torch
import torch.nn.functional as F

from distdl.utilities.torch import to_torch_pad


class DistributedPad(torch.nn.Module):
    """
    Pads the input with respect to it's global shape, as though it were on a single partition.
    This does not maintain balance of the partitions.

    Parameters
    ----------
    P_x :
        Cartesian Partition of input tensor.
    pad :
        NumPy ndarray (or list of tuples) of shape num_dimensions x 2.
        For each dimension, the inner list provides the left and right global padding to apply to the input.
    mode :
        'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
    value :
        fill value for 'constant' padding. Default: 0
    """

    def __init__(self, P_x, pad, mode='constant', value=0):

        super(DistributedPad, self).__init__()

        self.mode = mode
        self.value = value

        # For each dimension, determine if left or right side should be padded
        should_pad_left = [k == 0 for k in P_x.index]
        should_pad_right = [k == d-1 for k, d in zip(P_x.index, P_x.shape)]
        should_pad = np.stack((should_pad_left, should_pad_right), axis=1)
        self.local_pad = np.where(should_pad, pad, 0)
        self.torch_pad = to_torch_pad(self.local_pad)

    def forward(self, input):
        return F.pad(input, self.torch_pad, mode=self.mode, value=self.value)
