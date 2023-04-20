from mpi4py import MPI

from torchfire import torchfireRunWorker

def run(factory_functions: dict, ensemble_comm, local_comm) -> None:
    torchfireRunWorker(factory_functions, {}, ensemble_comm, local_comm)
