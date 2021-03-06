import numpy as np
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.mixins.halo_mixin import HaloMixin
from distdl.nn.mixins.pooling_mixin import PoolingMixin
from distdl.utilities.debug import print_sequential


class MockPoolLayer(HaloMixin, PoolingMixin):
    pass


P_world = MPIPartition(MPI.COMM_WORLD)
ranks = np.arange(P_world.size)

shape = [1, 1, 4]
P_size = np.prod(shape)
use_ranks = ranks[:P_size]

P = P_world.create_subpartition(use_ranks)
P_x = P.create_cartesian_subpartition(shape)
rank = P_x.rank
cart_comm = P_x._comm

layer = MockPoolLayer()

if P_x.active:
    x_global_shape = np.array([1, 1, 10])
    kernel_size = np.array([2])
    stride = np.array([2])
    padding = np.array([0])
    dilation = np.array([1])

    halo_shape, recv_buffer_shape, send_buffer_shape, needed_ranges = \
        layer._compute_exchange_info(x_global_shape,
                                     kernel_size,
                                     stride,
                                     padding,
                                     dilation,
                                     P_x.active,
                                     P_x.shape,
                                     P_x.index)

    print_sequential(cart_comm, f'rank = {rank}:\nhalo_shape =\n{halo_shape}\n\
recv_buffer_shape =\n{recv_buffer_shape}\nsend_buffer_shape =\n{send_buffer_shape}\nneeded_ranges =\n{needed_ranges}')
