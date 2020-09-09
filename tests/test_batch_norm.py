import numpy as np
import pytest

import distdl

parametrizations = []

parametrizations.append(
    pytest.param(
        np.arange(0, 2), [1, 1, 2],  # P_x_ranks, P_x_shape
        [0], [1, 1, 1],  # P_sum_ranks, P_sum_shape
        2,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-1d",
        marks=[pytest.mark.mpi(min_size=2)]
        )
    )

# num_dimensions,
# num_features, eps=1e-05, momentum=0.1, affine=True,
# track_running_statistics=True


@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "P_sum_ranks, P_sum_shape,"
                         "comm_split_fixture",
                         parametrizations,
                         indirect=["comm_split_fixture"])
def test_conv_class_selection(barrier_fence_fixture,
                              P_x_ranks, P_x_shape,
                              P_sum_ranks, P_sum_shape,
                              comm_split_fixture
                              ):

    from distdl.backends.mpi.partition import MPIPartition

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)
    P_sum_base = P_world.create_partition_inclusive(P_sum_ranks)
    P_sum = P_sum_base.create_cartesian_topology_partition(P_sum_shape)

    layer = distdl.nn.DistributedBatchNorm(P_x,
                                           P_sum,
                                           num_dimensions=3,
                                           num_features=3)
