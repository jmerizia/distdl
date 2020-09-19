import numpy as np
import pytest
import torch

import distdl
from distdl.utilities.torch import zero_volume_tensor
# from distdl.utilities.debug import print_sequential

ERROR_THRESHOLD = 1e-4
parametrizations_affine = []
parametrizations_non_affine = []
parametrizations_running = []

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [2, 1, 2],  # P_x_ranks, P_x_shape,
        (4, 3, 10),  # input_shape
        3, 1e-05, 0.1, True,  # num_features, eps, momentum, affine,
        False,  # track_running_statistics
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-batch",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [1, 4, 1],  # P_x_ranks, P_x_shape,
        (4, 4, 10),  # input_shape
        4, 1e-03, 0.2, True,  # num_features, eps, momentum, affine,
        False,  # track_running_statistics
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-feature",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [1, 2, 2],  # P_x_ranks, P_x_shape,
        (7, 13, 11),  # input_shape
        13, 1e-05, 0.1, True,  # num_features, eps, momentum, affine,
        True,  # track_running_statistics
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-feature-track-running",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [2, 2],  # P_x_ranks, P_x_shape,
        (7, 13),  # input_shape
        13, 1e-05, 0.1, True,  # num_features, eps, momentum, affine,
        False,  # track_running_statistics
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-2d",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 8), [1, 2, 2, 2],  # P_x_ranks, P_x_shape,
        (7, 13, 11, 3),  # input_shape
        13, 1e-05, 0.1, True,  # num_features, eps, momentum, affine,
        False,  # track_running_statistics
        8,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-4d",
        marks=[pytest.mark.mpi(min_size=8)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 12), [1, 3, 2, 2],  # P_x_ranks, P_x_shape,
        (7, 13, 11, 3),  # input_shape
        13, 1e-05, 0.1, True,  # num_features, eps, momentum, affine,
        False,  # track_running_statistics
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-4d-many-ranks",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

parametrizations_non_affine.append(
    pytest.param(
        np.arange(0, 4), [1, 2, 2],  # P_x_ranks, P_x_shape,
        (7, 13, 11),  # input_shape
        13, 1e-05, 0.1, False,  # num_features, eps, momentum, affine,
        False,  # track_running_statistics
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-no-affine-feature",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_non_affine.append(
    pytest.param(
        np.arange(0, 4), [1, 2, 2],  # P_x_ranks, P_x_shape,
        (7, 13, 11),  # input_shape
        13, 1e-05, 0.1, False,  # num_features, eps, momentum, affine,
        True,  # track_running_statistics
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-no-affine-feature-track-running",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )


@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "input_shape,"
                         "num_features, eps, momentum, affine,"
                         "track_running_stats,"
                         "comm_split_fixture",
                         parametrizations_affine,
                         indirect=["comm_split_fixture"])
def test_batch_norm_with_training(barrier_fence_fixture,
                                  P_x_ranks, P_x_shape,
                                  input_shape,
                                  num_features, eps, momentum, affine,
                                  track_running_stats,
                                  comm_split_fixture):

    from distdl.backends.mpi.partition import MPIPartition

    torch.manual_seed(0)

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    num_dimensions = len(input_shape)
    P_in_out_base = P_world.create_partition_inclusive([0])
    P_in_out = P_in_out_base.create_cartesian_topology_partition([1] * num_dimensions)
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    # Create the input
    if P_world.comm.Get_rank() == 0:
        input_train = torch.rand(input_shape, dtype=torch.float32)
        input_eval = torch.rand(input_shape, dtype=torch.float32)
        exp = torch.rand(input_shape, dtype=torch.float32)
    else:
        input_train = zero_volume_tensor()
        input_eval = zero_volume_tensor()
        exp = zero_volume_tensor()

    # Create the sequential network
    layer = torch.nn.BatchNorm1d if len(input_shape) <= 3 else torch.nn.BatchNorm2d
    if P_world.comm.Get_rank() == 0:
        seq_net = torch.nn.Sequential(layer(num_features=num_features,
                                            eps=eps,
                                            momentum=momentum,
                                            affine=affine,
                                            track_running_stats=track_running_stats))
        seq_parameters = list(seq_net.parameters())
        seq_optim = torch.optim.Adam(seq_parameters, lr=1e-1)
    else:
        seq_optim = None
        seq_net = None

    # Train sequential network
    if P_world.comm.Get_rank() == 0:
        seq_net.train()
        seq_optim.zero_grad()
        seq_out1 = seq_net(input_train)
        seq_loss = ((seq_out1 - exp) ** 2).sum()
        seq_loss.backward()
        seq_optim.step()
    else:
        seq_out1 = None
        seq_loss = None

    # Evaluate sequential network
    if P_world.comm.Get_rank() == 0:
        seq_net.eval()
        seq_out2 = seq_net(input_eval)

    # Create distributed network
    dist_net = torch.nn.Sequential(distdl.nn.DistributedTranspose(P_in_out, P_x),
                                   distdl.nn.DistributedBatchNorm(P_x,
                                                                  num_features=num_features,
                                                                  eps=eps,
                                                                  momentum=momentum,
                                                                  affine=affine,
                                                                  track_running_stats=track_running_stats),
                                   distdl.nn.DistributedTranspose(P_x, P_in_out))
    dist_parameters = list(dist_net.parameters())
    dist_optim = torch.optim.Adam(dist_parameters, lr=1e-1)

    # Train distributed network
    dist_net.train()
    dist_optim.zero_grad()
    dist_out1 = dist_net(input_train)
    # assert dist_out1.shape == exp.shape
    dist_loss = ((dist_out1 - exp) ** 2).sum()
    assert dist_loss.requires_grad
    dist_loss.backward()
    dist_optim.step()

    # Evaluate distributed network
    dist_net.eval()
    dist_out2 = dist_net(input_eval)

    # Compare the distributed and sequential networks
    if P_world.comm.Get_rank() == 0:
        assert dist_out1.shape == seq_out1.shape
        assert torch.allclose(dist_out1, seq_out1, ERROR_THRESHOLD)
        assert dist_loss.shape == seq_loss.shape
        assert torch.allclose(dist_loss, seq_loss, ERROR_THRESHOLD)
        assert dist_out2.shape == seq_out2.shape
        assert torch.allclose(dist_out2, seq_out2, ERROR_THRESHOLD)


@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "input_shape,"
                         "num_features, eps, momentum, affine,"
                         "track_running_stats,"
                         "comm_split_fixture",
                         parametrizations_non_affine,
                         indirect=["comm_split_fixture"])
def test_batch_norm_no_training(barrier_fence_fixture,
                                P_x_ranks, P_x_shape,
                                input_shape,
                                num_features, eps, momentum, affine,
                                track_running_stats,
                                comm_split_fixture):

    from distdl.backends.mpi.partition import MPIPartition

    torch.manual_seed(0)

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    num_dimensions = len(input_shape)
    P_in_out_base = P_world.create_partition_inclusive([0])
    P_in_out = P_in_out_base.create_cartesian_topology_partition([1] * num_dimensions)
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    # Create the input
    if P_world.comm.Get_rank() == 0:
        input_eval = torch.rand(input_shape, dtype=torch.float32)
    else:
        input_eval = zero_volume_tensor()

    # Create the sequential network
    if P_world.comm.Get_rank() == 0:
        seq_net = torch.nn.Sequential(torch.nn.BatchNorm1d(num_features=num_features,
                                                           eps=eps,
                                                           momentum=momentum,
                                                           affine=affine,
                                                           track_running_stats=track_running_stats))
    else:
        seq_net = None

    # Evaluate sequential network
    if P_world.comm.Get_rank() == 0:
        seq_net.eval()
        seq_out = seq_net(input_eval)

    # Create distributed network
    dist_net = torch.nn.Sequential(distdl.nn.DistributedTranspose(P_in_out, P_x),
                                   distdl.nn.DistributedBatchNorm(P_x,
                                                                  num_features=num_features,
                                                                  eps=eps,
                                                                  momentum=momentum,
                                                                  affine=affine,
                                                                  track_running_stats=track_running_stats),
                                   distdl.nn.DistributedTranspose(P_x, P_in_out))

    # Evaluate distributed network
    dist_net.eval()
    dist_out = dist_net(input_eval)

    # Compare the distributed and sequential networks
    if P_world.comm.Get_rank() == 0:
        assert dist_out.shape == seq_out.shape
        assert torch.allclose(dist_out, seq_out, ERROR_THRESHOLD)
