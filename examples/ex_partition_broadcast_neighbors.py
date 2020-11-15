import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.utilities.debug import print_sequential

# Set up partitions
P_world = MPIPartition(MPI.COMM_WORLD)
P_base = P_world.create_partition_inclusive(np.arange(27))
P_cart = P_base.create_cartesian_topology_partition([3, 3, 3])

if not P_cart.active:
    quit()

data = np.ones((2, 2), dtype=np.int) * P_cart.rank
neighbor_data = P_cart.broadcast_to_neighbors(data)
neighbor_ranks = P_cart.neighbor_ranks(P_cart.rank)

for dim in range(3):
    left_data, right_data = neighbor_data[dim]
    left_rank, right_rank = neighbor_ranks[dim]

    if left_rank == MPI.PROC_NULL:
        assert left_data is None
    else:
        assert (left_data == left_rank).all()

    if right_rank == MPI.PROC_NULL:
        assert right_data is None
    else:
        assert (right_data == right_rank).all()
