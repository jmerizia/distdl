import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.mixins.conv_mixin import ConvMixin
from distdl.nn.mixins.halo_mixin import HaloMixin
from distdl.nn.padnd import PadNd
from distdl.utilities.debug import print_sequential
from distdl.utilities.misc import DummyContext
from distdl.utilities.slicing import compute_subshape


class MockConvLayer(HaloMixin, ConvMixin):
    pass


torch.set_printoptions(linewidth=200)

P_world = MPIPartition(MPI.COMM_WORLD)
ranks = np.arange(P_world.size)

shape = [1, 1, 2, 2]
P_size = np.prod(shape)
use_ranks = ranks[:P_size]

P_x_base = P_world.create_partition_inclusive(use_ranks)
P_x = P_x_base.create_cartesian_topology_partition(shape)
rank = P_x.rank
cart_comm = P_x._comm

x_global_shape = np.array([1, 1, 10, 12])

if P_x.active:
    mockup_conv_layer = MockConvLayer()
    kernel_size = [1, 1, 3, 3]
    stride = [1, 1, 1, 1]
    padding = [0, 0, 0, 0]
    dilation = [1, 1, 1, 1]

    exchange_info = mockup_conv_layer._compute_exchange_info(x_global_shape,
                                                             kernel_size,
                                                             stride,
                                                             padding,
                                                             dilation,
                                                             P_x.active,
                                                             P_x.shape,
                                                             P_x.index)
    halo_shape = exchange_info[0]
    recv_buffer_shape = exchange_info[1]
    send_buffer_shape = exchange_info[2]

    x_local_shape = compute_subshape(P_x.shape,
                                     P_x.index,
                                     x_global_shape)

    value = (1 + rank) * (10 ** rank)
    a = np.full(shape=x_local_shape, fill_value=value, dtype=float)

    forward_input_padnd_layer = PadNd(halo_shape.astype(int), value=0, partition=P_x)
    adjoint_input_padnd_layer = PadNd(halo_shape.astype(int), value=value, partition=P_x)
    t = torch.tensor(a, requires_grad=True)
    t_forward_input = forward_input_padnd_layer.forward(t)
    t_adjoint_input = adjoint_input_padnd_layer.forward(t)

    halo_layer = HaloExchange(P_x, halo_shape, recv_buffer_shape, send_buffer_shape)

    print_sequential(cart_comm, f'rank = {rank}, t_forward_input =\n{t_forward_input.int()}')

    ctx = DummyContext()
    t_forward_exchanged = halo_layer(t_forward_input)

    print_sequential(cart_comm, f'rank = {rank}, t_forward_exchanged =\n{t_forward_input.int()}')

    print_sequential(cart_comm, f'rank = {rank}, t_adjoint_input =\n{t_adjoint_input.int()}')

    t_forward_exchanged.backward(t_adjoint_input)

    print_sequential(cart_comm, f'rank = {rank}, t_adjoint_exchanged =\n{t_adjoint_input.int()}')
