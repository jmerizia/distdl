import numpy as np
import pytest
from adjoint_test import check_adjoint_test_tight

adjoint_parametrizations = []

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_topo
        [1, 1, 7, 9],  # global_tensor_size
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-weird_1",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 9), [1, 1, 3, 3],  # P_x_ranks, P_x_topo
        [1, 1, 18, 23],  # global_tensor_size
        9,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-weird_2",
        marks=[pytest.mark.mpi(min_size=9)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_topo
        [1, 1, 8, 8],  # global_tensor_size
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-no_comm",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_topo,"
                         "global_tensor_size,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_average_pooling_adjoint_input(barrier_fence_fixture,
                                       comm_split_fixture,
                                       P_x_ranks, P_x_topo,
                                       global_tensor_size):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.pooling import DistributedAvgPool2d
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_topo)

    global_tensor_sizes = np.asarray(global_tensor_size)

    layer = DistributedAvgPool2d(P_x,
                                 kernel_size=[2, 2],
                                 stride=[2, 2])

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              global_tensor_sizes)
        x = torch.tensor(np.random.randn(*input_tensor_sizes))
    x.requires_grad = True

    y = layer(x)

    dy = NoneTensor()
    if P_x.active:
        dy = torch.Tensor(np.random.randn(*y.shape))

    y.backward(dy)
    dx = x.grad

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, x, dx, y, dy)