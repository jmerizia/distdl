import numpy as np
import torch
import random
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.batch_norm import DistributedBatchNorm
from distdl.nn.transpose import DistributedTranspose
from distdl.utilities.debug import print_sequential
# from distdl.nn.transpose import DistributedTranspose
from distdl.utilities.slicing import compute_subshape, assemble_slices, \
    compute_start_index, compute_stop_index
from distdl.utilities.torch import zero_volume_tensor


# Set up partitions
P_world = MPIPartition(MPI.COMM_WORLD)
P_world.comm.Barrier()

P_base = P_world.create_partition_inclusive([0, 1, 2, 3])
P_sum_base = P_world.create_partition_inclusive([0])
P_output_base = P_world.create_partition_inclusive([0])

P_in = P_output_base.create_cartesian_topology_partition([1, 1, 1])
P_x = P_base.create_cartesian_topology_partition([2, 1, 2])
P_sum = P_sum_base.create_cartesian_topology_partition([1, 1, 1])
P_y = P_base.create_cartesian_topology_partition([2, 1, 2])
P_out = P_output_base.create_cartesian_topology_partition([1, 1, 1])

# Get input onto all ranks
if P_world.comm.Get_rank() == 0:
    input = np.random.rand(4, 3, 10)
    input = torch.tensor(input, dtype=torch.float32)
else:
    input = None
input = P_world.comm.scatter([input]*4, root=0)
input_shape = input.shape

# Run sequential batch norm on rank 0 as reference
if P_world.comm.Get_rank() == 0:
    s_batch_norm = torch.nn.BatchNorm1d(num_features=3, track_running_stats=False)
    y1 = s_batch_norm(input)
    print_sequential(P_world.comm, f'reference: {y1}, {y1.shape}')
else:
    print_sequential(P_world.comm, None)

# Partition the input for distributed batch norm
if P_x.active:
    a = compute_start_index(P_x.shape, P_x.index, input_shape)
    b = compute_stop_index(P_x.shape, P_x.index, input_shape)
    slices = assemble_slices(a, b)
    # print_sequential(P_world.comm, f'{slices}')
    x = input[slices[0], slices[1], slices[2]]
else:
    x = zero_volume_tensor()

# print_sequential(P_world.comm, f'x = {x.shape}\n{x.detach().numpy()}')

# run the distributed bn layer
d_transpose_in = DistributedTranspose(P_y, P_out)
d_batch_norm = DistributedBatchNorm(P_x, P_sum, P_y, num_features=3)
d_transpose_out = DistributedTranspose(P_y, P_out)
y2 = d_transpose_in(x)
y2 = d_batch_norm(y2)
y2 = d_transpose_out(y2)

if P_world.comm.Get_rank() == 0:
    avg_error = (y1 - y2).sum() / (y1.shape[0] + y1.shape[1] * y1.shape[2])
    print_sequential(P_world.comm, f'distributed: {y2}, {y2.shape}, average_error = {avg_error}')
else:
    print_sequential(P_world.comm, None)

print_sequential(P_world.comm, 'done')
quit()

# run the sequential bn layer
if P_x.comm.Get_rank() == 0:
    # s_batch_norm = torch.nn.BatchNorm1d(1)
    # y1 = s_batch_norm(x)
    y1 = torch.sum(input, dim=0, keepdim=True) 
    y1 += torch.tensor(np.zeros((4, 10)), dtype=torch.float32)
    y1 /= (input_shape[0] * input_shape[2])

# print the results
if P_x.comm.Get_rank() == 0:
    print_sequential(P_world.comm, f'distributed bn : {y2.shape}\n{y2.detach().numpy()}')
    print_sequential(P_world.comm, f'reference bn : {y1.shape}\n{y1.detach().numpy()}')
else:
    print_sequential(P_world.comm, None)
    print_sequential(P_world.comm, None)
