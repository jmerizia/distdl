import numpy as np
import pytest
from adjoint_test import check_adjoint_test_tight

params = []

# params.append(
#     pytest.param(
#         np.arange(0, 3), [1, 1, 3, 1],  # P_x_ranks, P_x_shape
#         [1, 5, 3, 3],  # x_global_shape
#         [3, 3],  # kernel_size
#         [2, 2],  # padding
#         [1, 1],  # stride
#         [1, 1],  # dilation
#         3,  # passed to comm_split_fixture, required MPI ranks
#         id="distributed",
#         marks=[pytest.mark.mpi(min_size=3)]
#     )
# )


@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "x_global_shape,"
                         "kernel_size,"
                         "padding,"
                         "stride,"
                         "dilation,"
                         "comm_split_fixture",
                         params,
                         indirect=["comm_split_fixture"])
def test_conv2d_versus_pytorch(barrier_fence_fixture,
                               comm_split_fixture,
                               P_x_ranks, P_x_shape,
                               x_global_shape,
                               kernel_size,
                               padding,
                               stride,
                               dilation):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.conv_feature import DistributedFeatureConv2d
    from distdl.nn.transpose import DistributedTranspose
    from distdl.utilities.torch import zero_volume_tensor
    from torch.nn import Conv2d

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_root_base = P_world.create_partition_inclusive([0])
    P_root = P_root_base.create_cartesian_topology_partition([1]*len(P_x_shape))
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    x_global_shape = np.asarray(x_global_shape)

    # Create layers
    seq_layer = Conv2d(in_channels=x_global_shape[1],
                       out_channels=10,
                       kernel_size=kernel_size,
                       padding=padding,
                       stride=stride,
                       dilation=dilation,
                       bias=False)
    dist_layer = DistributedFeatureConv2d(P_x,
                                          in_channels=x_global_shape[1],
                                          out_channels=10,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          stride=stride,
                                          dilation=dilation,
                                          bias=False)
    scatter = DistributedTranspose(P_root, P_x)
    gather = DistributedTranspose(P_x, P_root)

    # Create Input
    x = torch.rand(tuple(x_global_shape))
    x.requires_grad = True

    # Run the layers
    y = seq_layer(x)
    y_hat = gather(dist_layer(scatter(x)))
    print(y_hat.shape)
    assert False

    print(y - y_hat)
    assert False

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
