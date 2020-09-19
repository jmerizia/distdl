import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.batch_norm import DistributedBatchNorm
from distdl.nn.transpose import DistributedTranspose
from distdl.utilities.debug import print_sequential
from distdl.utilities.torch import zero_volume_tensor


# set up partitions
P_world = MPIPartition(MPI.COMM_WORLD)
P_world.comm.Barrier()

P_base = P_world.create_partition_inclusive(np.arange(8))
P_sum_base = P_world.create_partition_inclusive([0])
P_output_base = P_world.create_partition_inclusive([0])

P_in = P_output_base.create_cartesian_topology_partition([1, 1, 1])
P_x = P_base.create_cartesian_topology_partition([4, 1, 2])
P_sum = P_sum_base.create_cartesian_topology_partition([1, 1, 1])
P_out = P_output_base.create_cartesian_topology_partition([1, 1, 1])

# set a random input in rank 0
if P_world.rank == 0:
    input = np.random.rand(4, 3, 10)
    input = torch.tensor(input, dtype=torch.float32)
else:
    input = zero_volume_tensor()

# run sequential batch norm on rank 0 as reference
if P_world.rank == 0:
    s_batch_norm = torch.nn.BatchNorm1d(num_features=3, track_running_stats=True)
    y1 = s_batch_norm(input)
    print_sequential(P_world.comm, f'reference: {y1}, {y1.shape}')
else:
    print_sequential(P_world.comm, None)

# run the distributed bn layer
d_transpose_in = DistributedTranspose(P_in, P_x)
d_batch_norm = DistributedBatchNorm(P_x, P_sum, num_dimensions=3, num_features=3)
d_transpose_out = DistributedTranspose(P_x, P_out)
y2 = d_transpose_in(input)
y2 = d_batch_norm(y2)
y2 = d_transpose_out(y2)

if P_world.rank == 0:
    avg_error = (y1 - y2).sum() / (y1.shape[0] * y1.shape[1] * y1.shape[2])
    print_sequential(P_world.comm, f'distributed: {y2}, {y2.shape}, average_error = {avg_error}')
else:
    print_sequential(P_world.comm, None)
