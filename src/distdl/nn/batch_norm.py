import torch

from distdl.nn.sum_reduce import SumReduce
from distdl.nn.broadcast import Broadcast
from distdl.nn.module import Module
from distdl.backends.mpi.tensor_comm import compute_global_tensor_shape


class DistributedBatchNorm(Module):
    r"""A distributed batch norm layer.

    Applies Batch Normalization using mini-batch statistics.
    This layer is a distributed and generalized version of the PyTorch BatchNormNd layers.
    Currently, parallelism is supported in all dimensions except the feature dimension (dimension 2).

    Parameters
    ----------
    P_x :
        Partition of input tensor.
    P_sum :
        An internal paritition used to compute mini-batch statistics,
        store running statistics, and store affine weights.
    num_dimensions :
        The number of dimensions in the input shape.
    num_features :
        Number of features in the input.
        For exmaple, this should equal C in an input of shape (N, C, L).
    eps : optional
        A value added to the denominator for numerical stability.
        Default is 1e-5.
    momentum : optional
        The value used for the running_mean and running_var computation.
        Can be set to None for cumulative moving average (i.e. simple average).
        Default is 0.1.
    affine : optional
        a boolean value that when set to True, this module has learnable affine parameters.
        Default is True.
    track_running_statistics : optional
        a boolean value that when set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics and uses batch statistics
        instead in both training and eval modes if the running mean and variance are None.
        Default is True.
    """

    def __init__(self, P_x, P_sum, num_dimensions,
                 num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_statistics=True):
        super(DistributedBatchNorm, self).__init__()
        assert num_dimensions >= 2
        self.P_x = P_x
        self.P_sum = P_sum
        self.num_dimensions = num_dimensions
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_statistics = track_running_statistics
        self.inputs_seen = 0  # Note: this is used for cumulative moving average

        internal_data_shape = [1 if i != 1 else self.num_features for i in range(self.num_dimensions)]
        if self.track_running_statistics:
            self.running_mean = torch.zeros(internal_data_shape)
            self.running_var = torch.ones(internal_data_shape)
        else:
            self.running_mean = None
            self.running_var = None

        self.sr1 = SumReduce(P_x, P_sum)
        self.sr2 = SumReduce(P_x, P_sum)
        self.bc1 = Broadcast(P_sum, P_x)
        self.bc2 = Broadcast(P_sum, P_x)

        if self.affine:
            self.gamma = torch.nn.Parameter(torch.ones(internal_data_shape))
            self.beta = torch.nn.Parameter(torch.zeros(internal_data_shape))

    def _compute_mean(self, x, feature_volume):
        '''
        Compute global mean given the input.
        Ensures all ranks have the mean tensor.
        '''
        for dim in range(self.num_dimensions):
            if dim != 1:
                x = x.sum(dim, keepdim=True)
        x = self.sr1(x)
        x /= feature_volume
        x = self.bc1(x)
        return x

    def _compute_var(self, input, mean, feature_volume):
        '''
        Compute global variance given the input and global mean.
        Ensures all ranks have the variance tensor.
        '''
        x = (input - mean) ** 2
        for dim in range(self.num_dimensions):
            if dim != 1:
                x = x.sum(dim, keepdim=True)
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
        var = self._compute_var(input, mean, feature_volume)

        # update running statistics
        if self.track_running_statistics and self.training:
            if self.momentum:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            else:
                # use a cumulative moving average instead
                self.running_mean = (self.running_mean + self.inputs_seen * mean) / (self.inputs_seen + 1)
                self.running_var = (self.running_var + self.inputs_seen * var) / (self.inputs_seen + 1)
                self.inputs_seen += 1

        # normalize
        x = (input - mean) / (var + self.eps) ** 0.5

        # scale and shift
        if self.affine:
            x = self.gamma * x + self.beta

        return x
