# import numpy as np
import torch
# from mpi4py import MPI

# from distdl.nn.broadcast import Broadcast
# from distdl.utilities.debug import print_sequential
from distdl.nn.module import Module


class DistributedBatchNorm(Module):

    def __init__(self, P_x, eps=1e-5, momentum=0.1, affine=True):

        super(DistributedBatchNorm, self).__init__()
        self.P_x = P_x
        self.eps = eps
        self.size = P_x.comm.Get_size()
        self.rank = P_x.comm.Get_rank()

        # self.bn = torch.nn.BatchNorm2d(num_features=num_features,
        #                                eps=eps,
        #                                momentum=momentum,
        #                                affine=affine,
        #                                track_running_stats=False)

    def _distributed_mean(self, input):
        mean = torch.mean(input).item()
        volume = input.nelement()

        x = self.P_x.comm.gather([mean, volume], root=0)

        if self.rank == 0:
            x = [sum(m * v for m, v in x)/sum(v for m, v in x)] * self.size
        else:
            x = None

        x = self.P_x.comm.scatter(x, root=0)
        return x

    def _distributed_variance(self, input, overall_mean):
        volume = input.nelement()
        x = (input - overall_mean) ** 2
        x = torch.sum(x).item()

        x = self.P_x.comm.gather([x, volume], root=0)

        if self.rank == 0:
            x = [sum(t for t, v in x)/sum(v for t, v in x)] * self.size
        else:
            x = None

        x = self.P_x.comm.scatter(x, root=0)
        return x

    def forward(self, input):

        if not self.P_x.active:
            return input.clone()

        overall_mean = self._distributed_mean(input)
        overall_variance = self._distributed_variance(input, overall_mean)

        x = input - overall_mean
        x = x / ((overall_variance + self.eps) ** 0.5)

        return x
