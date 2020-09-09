# import numpy as np
import torch
# from mpi4py import MPI

# from distdl.nn.broadcast import Broadcast
from distdl.nn.sum_reduce import SumReduce
from distdl.nn.broadcast import Broadcast
from distdl.utilities.debug import print_sequential
# from distdl.utilities.torch import zero_volume_tensor
from distdl.nn.module import Module
from distdl.backends.mpi.tensor_comm import compute_global_tensor_shape


class DistributedBatchNorm(Module):

    def __init__(self, P_x, P_sum, P_y,
                 num_features,
                 eps=1e-5, momentum=0.1, affine=True,
                 track_running_statistics=True):
        super(DistributedBatchNorm, self).__init__()
        self.P_x = P_x
        self.P_sum = P_sum
        self.P_y = P_y
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # self.running_mean = torch.tensor(np.)

        self.sr1 = SumReduce(P_x, P_sum)
        self.sr2 = SumReduce(P_x, P_sum)
        self.bc1 = Broadcast(P_sum, P_x)
        self.bc2 = Broadcast(P_sum, P_x)

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1)))
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1)))

    def _compute_mean(self, x, feature_volume):
        '''
        Compute global mean given the input.
        Ensures all ranks have the mean tensor.
        '''
        x = torch.sum(x, 0, keepdim=True)
        x = torch.sum(x, 2, keepdim=True)
        x = self.sr1(x)
        x /= feature_volume
        x = self.bc1(x)
        return x

    def _compute_variance(self, input, mean, feature_volume):
        '''
        Compute global variance given the input and global mean.
        Ensures all ranks have the variance tensor.
        '''
        x = (input - mean) ** 2
        x = torch.sum(x, 0, keepdim=True)
        x = torch.sum(x, 2, keepdim=True)
        x = self.sr2(x)
        x /= feature_volume
        x = self.bc2(x)
        return x

    def forward(self, input):
        assert input.shape[1] == self.num_features

        # compute the volume of a feature
        global_shape = compute_global_tensor_shape(input, self.P_x)
        feature_volume = global_shape[0] * global_shape[2]

        # mini-batch statistics
        mean = self._compute_mean(input, feature_volume)
        variance = self._compute_variance(input, mean, feature_volume)

        # update running statistics
        # self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
        # self.running_variance = (1 - self.momentum) * self.running_variance + self.momentum * variance

        # normalize
        x = (input - mean) / (variance + self.eps) ** 0.5

        # scale and shift
        x = self.gamma * x + self.beta

        return x
