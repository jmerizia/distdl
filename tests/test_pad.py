import pytest
import torch
import numpy as np

parametrizations = []

# Basic symmetric padding on all dimensions
parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 2, 2],  # P_x_ranks, P_x_shape
        [1, 2, 3],  # global_input_shape
        torch.float32,  # dtype
        [[0, 0], [1, 1], [1, 1]],  # pad
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-padding-1",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# No pad
parametrizations.append(
    pytest.param(
        np.arange(0, 3), [1, 3],  # P_x_ranks, P_x_shape
        [3, 3],  # global_input_shape
        torch.float64,  # dtype
        [[0, 0], [0, 0]],  # pad
        3,  # passed to comm_split_fixture, required MPI ranks
        id="positive_padding-float64",
        marks=[pytest.mark.mpi(min_size=3)]
        )
    )

# Asymmetric padding, and 3D input
parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 2, 2],  # P_x_ranks, P_x_shape
        [3, 5, 3],  # global_input_shape
        torch.float32,  # dtype
        [[1, 0], [0, 2], [0, 0]],  # pad
        4,  # passed to comm_split_fixture, required MPI ranks
        id="nonnegative_padding-float32",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# 1D input
parametrizations.append(
    pytest.param(
        np.arange(0, 2), [2],  # P_x_ranks, P_x_shape
        [5],  # global_input_shape
        torch.float32,  # dtype
        [[1, 2]],  # pad
        2,  # passed to comm_split_fixture, required MPI ranks
        id="nonnegative_padding-float32",
        marks=[pytest.mark.mpi(min_size=2)]
        )
    )

# 4D input
parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
        [3, 3, 6, 6],  # global_input_shape
        torch.float32,  # dtype
        [[1, 1], [0, 0], [3, 3], [1, 2]],  # pad
        4,  # passed to comm_split_fixture, required MPI ranks
        id="nonnegative_padding-float32",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )


@pytest.mark.parametrize("P_x_ranks,"
                         "P_x_shape,"
                         "global_input_shape,"
                         "dtype,"
                         "pad,"
                         "comm_split_fixture",
                         parametrizations,
                         indirect=["comm_split_fixture"])
def test_padnd_adjoint(barrier_fence_fixture,
                       comm_split_fixture,
                       P_x_ranks, P_x_shape,
                       global_input_shape,
                       dtype,
                       pad):

    import numpy as np
    import torch
    import torch.nn.functional as F

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.pad import DistributedPad
    from distdl.nn.transpose import DistributedTranspose
    from distdl.utilities.torch import zero_volume_tensor
    from distdl.utilities.torch import to_torch_pad

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create partitions
    P_root_base = P_world.create_partition_inclusive(np.arange(1))
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_root = P_root_base.create_cartesian_topology_partition([1]*len(P_x_shape))
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    # Create the distributed layers
    scatter = DistributedTranspose(P_root, P_x)
    dist_pad = DistributedPad(P_x, pad)
    gather = DistributedTranspose(P_x, P_root)

    # Create the input
    if P_root.active:
        dist_x = torch.randn(*global_input_shape).to(dtype)
        seq_x = dist_x.detach().clone()
        dist_x.requires_grad = True
        seq_x.requires_grad = True
    else:
        dist_x = zero_volume_tensor(requires_grad=True)

    # Check the forward pass
    dist_y = gather(dist_pad(scatter(dist_x)))
    if P_root.active:
       seq_y = F.pad(seq_x, to_torch_pad(pad))
       assert dist_y.dtype == seq_y.dtype
       assert dist_y.shape == seq_y.shape
       assert torch.allclose(dist_y, seq_y)

    # Check the backward pass
    dy = torch.zeros(*dist_y.shape, requires_grad=True)
    dist_y.backward(dy)
    if P_root.active:
        seq_y = seq_y.backward(dy)
        seq_dx = seq_x.grad
        dist_dx = dist_x.grad
        assert seq_dx.dtype == dist_dx.dtype
        assert seq_dx.shape == dist_dx.shape
        assert torch.allclose(seq_dx, dist_dx)

    P_world.deactivate()
