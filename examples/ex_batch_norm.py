import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.batch_norm import DistributedBatchNorm
from distdl.utilities.debug import print_sequential
# from distdl.utilities.slicing import compute_subshape
# from distdl.utilities.torch import zero_volume_tensor
from distdl.nn.transpose import DistributedTranspose

P_world = MPIPartition(MPI.COMM_WORLD)
P_world.comm.Barrier()

P_base = P_world.create_partition_inclusive([0, 1, 2, 3])

P_0 = P_base.create_partition_inclusive([0])
P_root = P_0.create_cartesian_topology_partition([1, 1, 1])

P_in = P_world.create_partition_inclusive([0, 1, 2, 3])
P_out = P_world.create_partition_inclusive([0])

P_x = P_in.create_cartesian_topology_partition([1, 1, 4])

if P_x.comm.Get_rank() == 0:
    x = torch.tensor(np.random.rand(1, 1, 10), dtype=torch.float32)
    x = [x] * P_x.comm.Get_size()
else:
    x = None

x = P_x.comm.scatter(x, root=0)

# if P_x.active:
#     x_local_shape = compute_subshape(P_x.shape,
#                                      P_x.index,
#                                      x)
#     a, b, c = x_local_shape
#     x = torch.tensor(np.random.rand(a, b, c), dtype=torch.float32)
# else:
#     x = zero_volume_tensor()

print_sequential(P_world.comm, f'x in rank {P_x.comm.Get_rank()} = {x.detach().numpy()}')

# run the distributed bn layer
d_transpose1 = DistributedTranspose(P_root, P_x)
d_batch_norm = DistributedBatchNorm(P_x)
d_transpose2 = DistributedTranspose(P_x, P_root)
y2 = d_transpose2(d_batch_norm(d_transpose1(x)))

# run the sequential bn layer
if P_x.comm.Get_rank() == 0:
    s_batch_norm = torch.nn.BatchNorm1d(1)
    y1 = s_batch_norm(x)

# print the results
if P_x.comm.Get_rank() == 0:
    print_sequential(P_world.comm, f'distributed bn : {y2.detach().numpy()}')
    print_sequential(P_world.comm, f'reference bn : {y1.detach().numpy()}')
else:
    print_sequential(P_world.comm, None)
    print_sequential(P_world.comm, None)
