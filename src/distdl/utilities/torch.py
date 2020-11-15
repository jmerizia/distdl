import torch
import numpy as np


def zero_volume_tensor(b=None, dtype=None, requires_grad=False):

    if dtype is None:
        dtype = torch.get_default_dtype()

    if b is None:
        return torch.empty((0,), dtype=dtype, requires_grad=requires_grad)

    return torch.empty((b, 0), dtype=dtype, requires_grad=requires_grad)


def to_torch_pad(pad):
    """
    Accepts a NumPy ndarray describing a pad, and produces the torch F.pad format.
    The shape of `pad' should be dims by 2.
    """
    return tuple(np.array(list(reversed(pad)), dtype=int).flatten())


class TensorStructure:
    """ Light-weight class to store and compare basic structure of Torch tensors.

    """

    def __init__(self, tensor=None, shape=None):

        self.shape = None
        self.dtype = None
        self.requires_grad = None

        if tensor is not None:
            self.fill_from_tensor(tensor)
        elif shape is not None:
            self.shape = shape

    def fill_from_tensor(self, tensor):

        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.requires_grad = tensor.requires_grad

    def __eq__(self, other):

        return ((self.shape == other.shape) and
                (self.dtype == other.dtype) and
                (self.requires_grad == other.requires_grad))
