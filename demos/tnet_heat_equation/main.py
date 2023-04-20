from mpi4py import MPI

from factory_functions import getFactoryFunctions
import root_logic
import worker_logic

world_comm = MPI.COMM_WORLD
rank = world_comm.Get_rank()
fact_funcs = getFactoryFunctions()
local_comm = world_comm.Split(rank, rank)

# We could divide the processes in such a way so as to
# use multiple ranks per Firedrake simulation in addition
# to parallelizing over the batch. The communicator that
# will communicate between pytorch (on rank 0) and the
# Firedrake functions will be called "ensemble_comm".
# For now, let's use COMM_WORLD. 
ensemble_comm = world_comm

if rank == 0:
    root_logic.run(fact_funcs, ensemble_comm)
else:
    # Here we split the global communicator into smaller comms for Firedrake. 
    # Replace this splitting with something more complicated if you want
    # more than one process per Firedrake simulation (worker).
    worker_logic.run(fact_funcs, ensemble_comm, local_comm)
