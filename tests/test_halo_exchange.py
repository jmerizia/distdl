import numpy as np
import pytest
from adjoint_test import check_adjoint_test_tight

from distdl.nn.halo_mixin import HaloMixin


class MockupConvLayer(HaloMixin):

    # These mappings come from basic knowledge of convolutions
    def _compute_min_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take stride and dilation into account
        # padding might also not be correct in these cases...
        kernel_offsets = (kernel_sizes - 1) / 2

        # for even sized kernels, always shortchange the left side
        kernel_offsets[kernel_sizes % 2 == 0] -= 1

        bases = idx + kernel_offsets - pads
        return bases - kernel_offsets

    def _compute_max_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take stride and dilation into account
        # padding might also not be correct in these cases...
        kernel_offsets = (kernel_sizes - 1) / 2

        bases = idx + kernel_offsets - pads
        return bases + kernel_offsets


class MockupPoolingLayer(HaloMixin):

    def _compute_min_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take dilation and padding into account
        return strides * idx + 0

    def _compute_max_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take dilation and padding into account
        return strides * idx + kernel_sizes - 1


adjoint_parametrizations = []

# Main functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 9), [1, 1, 3, 3],  # P_x_ranks, P_x_topo
        [1, 1, 10, 7],  # global_tensor_sizes
        [1, 1, 3, 3],  # kernel_sizes
        [1, 1, 1, 1],  # strides
        [0, 0, 0, 0],  # pads
        [1, 1, 1, 1],  # dilations
        MockupConvLayer,  # MockupKernelStyle
        9,  # passed to comm_split_fixture, required MPI ranks
        id="conv-same_padding",
        marks=[pytest.mark.mpi(min_size=9)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 3), [1, 1, 3],  # P_x_ranks, P_x_topo
        [1, 1, 10],  # global_tensor_sizes
        [2],  # kernel_sizes
        [2],  # strides
        [0],  # pads
        [1],  # dilations
        MockupConvLayer,  # MockupKernelStyle
        3,  # passed to comm_split_fixture, required MPI ranks
        id="conv-same_padding",
        marks=[pytest.mark.mpi(min_size=3)]
        )
    )


@pytest.mark.parametrize("P_x_ranks, P_x_topo,"
                         "global_tensor_sizes,"
                         "kernel_sizes,"
                         "strides,"
                         "pads,"
                         "dilations,"
                         "MockupKernelStyle,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_halo_exchange_adjoint(barrier_fence_fixture,
                               comm_split_fixture,
                               P_x_ranks, P_x_topo,
                               global_tensor_sizes,
                               kernel_sizes, strides, pads, dilations,
                               MockupKernelStyle):
    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.halo_exchange import HaloExchange
    from distdl.nn.padnd import PadNd
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_topo)

    global_tensor_sizes = np.asarray(global_tensor_sizes)
    kernel_sizes = np.asarray(kernel_sizes)
    strides = np.asarray(strides)
    pads = np.asarray(pads)
    dilations = np.asarray(dilations)

    halo_sizes = None
    recv_buffer_sizes = None
    send_buffer_sizes = None
    if P_x.active:
        mockup_layer = MockupKernelStyle()
        exchange_info = mockup_layer._compute_exchange_info(global_tensor_sizes,
                                                            kernel_sizes,
                                                            strides,
                                                            pads,
                                                            dilations,
                                                            P_x.active,
                                                            P_x.dims,
                                                            P_x.coords)
        halo_sizes = exchange_info[0]
        recv_buffer_sizes = exchange_info[1]
        send_buffer_sizes = exchange_info[2]

    pad_layer = PadNd(halo_sizes, value=0, partition=P_x)
    halo_layer = HaloExchange(halo_sizes, recv_buffer_sizes, send_buffer_sizes,
                              P_x)

    x = NoneTensor()
    if P_x.active:
        in_subsizes = compute_subsizes(P_x.comm.dims,
                                       P_x.comm.Get_coords(P_x.rank),
                                       global_tensor_sizes)
        x = torch.tensor(np.random.randn(*in_subsizes))
    x.requires_grad = True
    x = pad_layer.forward(x)

    dy = NoneTensor()
    if P_x.active:
        dy = torch.tensor(np.random.randn(*x.shape))

    x_clone = x.clone()
    dy_clone = dy.clone()

    # x_clone is be modified in place by halo_layer, but we assign y to
    # reference it for clarity
    y = halo_layer(x_clone)

    # dy_clone is modified in place by halo_layer-adjoint, but we assign dx to
    # reference it for clarity
    y.backward(dy_clone)
    dx = dy_clone

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, x, dx, y, dy)