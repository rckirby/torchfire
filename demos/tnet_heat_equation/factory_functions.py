from typing import Callable

import firedrake
import numpy as np
import pandas as pd
import torch
from firedrake import (DirichletBC, FunctionSpace, Constant,
                       TestFunction, UnitSquareMesh, Function, solve, dx, grad, inner)

from torchfire import fd_to_torch

def load_data(name, device, target_shape=(-1,)):
    return torch.tensor(np.reshape(pd.read_csv(name).to_numpy(), target_shape)).to(device)

def factory_solveFiredrake(args: dict) -> tuple[str, Callable]:
    """
       This factory function returns 
    
       Args:
           args: a dictionary with arguments used to configure forward solve
               Should have the following members:
                 1. bc: Firedrake boundary condition object

       Returns:
           A callable that solves the forward problem, Poisson's equation here

    """
    def func(exp_u):
        y = Function(exp_u.function_space())
        v = TestFunction(exp_u.function_space())

        # Mesh must be specified here, otherwise the program hangs
        f = Constant(20.0, exp_u.function_space()) 
        F = inner(exp_u * grad(y), grad(v)) * dx - f * v * dx
        solve(F == 0, y, bcs=args['bc'])
        return y
    return "solveFiredrake", func

def factory_mcLoss(args: dict) -> tuple[str, Callable]:
    """
       This factory function returns the model constrained loss function
       as a Callable. The purpose of using a function factory 
       is to enable the MC loss to be configured based on args that 
       may change, reducing the need for code duplication and modification. 

       The most important part of this function is the definition of the 
       Firedrake mesh. Each definition of the mesh requires its own MPI communicator
       since Firedrake uses MPI internally. Otherwise, we will be unable to 
       achieve ensemble parallelism.
    
       Args:
           args: a dictionary with arguments used to configure loss function
               Should have the following members:
                 1. nx (int)
                 2. ny (int)
                 3. local_comm (MPI communicator)
                 4. num_observation (int)

       Returns:
           A callable that computes the MC loss for a single batch element

    """
    # Create the Firedrake structures. 
    #   It is VERY IMPORTANT to specify a communicator for the mesh.
    #   Firedrake assumes MPI_COMM_WORLD if not specified, which
    #   will result in incorrect results and no parallelism
    mesh = UnitSquareMesh(args['nx'], args['ny'], comm=args['local_comm'])
    V = FunctionSpace(mesh, "P", 1)
    bc = DirichletBC(V, 0, (1, 2, 3))

    # Use a factory method to return the solver function so
    # we can pass in boundary conditions. This
    # general approach, while slightly increasing complexity,
    # can greatly reduce the amount of code duplication needed to
    # modify physics parameters. 
    _, solveFiredrake = factory_solveFiredrake({'bc': bc})

    # create the torchfire solver function
    templates = (firedrake.Function(V),)
    torchfireSolver = fd_to_torch(solveFiredrake, templates, "FiredrakeSolver").apply

    # load observation indices from file 
    obs_indices = load_data('data/Observation_indices.csv', args['device'],
                            (args['num_observation'], -1))
    def func(u_obs_true, k):
        # solve forward problem with differentiable solver
        u = torchfireSolver(k)
        # extract solution at obs_indices
        u = u[obs_indices].squeeze()

        # compute L2 loss 
        loss = torch.mean(torch.square(u - u_obs_true))
        return loss    
    return "mcLoss", func

# Dictionary with string-function mapping so master process can
# tell worker processes when to create function instances and with what data.
# This design is useful because multiple factory functions may exist in
# more complicated physics problems with different sets of data. 
def getFactoryFunctions():
    factory_functions = {
        'mcLoss': factory_mcLoss
    }
    return factory_functions
