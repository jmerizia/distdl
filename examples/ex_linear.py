import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.linear import DistributedLinear
from distdl.utilities.debug import print_sequential
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import zero_volume_tensor

P_world = MPIPartition(MPI.COMM_WORLD)
P_world.comm.Barrier()

P_base = P_world.create_partition_inclusive([0, 1, 2, 3])

P_base_lo = P_base.create_partition_inclusive([0, 1])
P_base_hi = P_base.create_partition_inclusive([2, 3])

P_fc_in = P_base_lo.create_cartesian_topology_partition([1, 2])
P_fc_out = P_base_hi.create_cartesian_topology_partition([1, 2])
P_fc_mtx = P_base.create_cartesian_topology_partition([2, 2])

x_global_shape = np.array([1, 10])
if P_fc_in.active:
    x_local_shape = compute_subshape(P_fc_in.shape,
                                     P_fc_in.index,
                                     x_global_shape)
    x = torch.tensor(np.ones(shape=x_local_shape), dtype=torch.float32)
else:
    x = zero_volume_tensor()

print_sequential(P_world.comm, x.shape)

layer = DistributedLinear(P_fc_in,
                          P_fc_out,
                          P_fc_mtx,
                          10, 1)

print_sequential(P_world.comm, x)
y = layer(x)
print_sequential(P_world.comm, y)
