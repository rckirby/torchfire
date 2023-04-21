# This is where the magic happens
import firedrake_adjoint  # noqa
import numpy as np
import torch
from fecr import evaluate_primal, evaluate_pullback

from time import sleep

from mpi4py import MPI
from mpi_function import FLAGS, MpiReduceTorchFunction

def fd_to_torch(fd_callable, templates, classname):
    """Creates a subclass of torch.autograd.Function implementing
    the static forward and backward methods in terms of fecr.
    Args:
        fd_callable: Firedrake function to be executed during the forward pass
        templates: Templates for converting arrays to Firedrake types
        classname: String with the class' name
    Returns:
        subclass of torch.autograd.Function with methods `forward` and
        `backward`
    """

    def forward(ctx, *inputs):

        device = inputs[0].device
        np_inputs = [t.cpu().detach().numpy().astype(np.float64()) for t in inputs]
        np_output, fd_output, fd_input, tape = evaluate_primal(
            fd_callable, templates, *np_inputs)
        ctx.save_for_backward(*inputs)
        ctx.stuff = (np_output, fd_output, fd_input, tape)
        output = torch.tensor(np_output).to(device)
        return output

    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        device = inputs[0].device
        np_output, fd_output, fd_input, tape = ctx.stuff

        if len(list(grad_output.size())) > 0:
            g = grad_output.cpu().detach().numpy().astype(np.float64)
            vjp_out = evaluate_pullback(fd_output, fd_input, tape, g)
            t_output = [torch.tensor(t).to(device) for t in vjp_out]
        else:
            g = np.ones_like(grad_output)
            vjp_out = evaluate_pullback(fd_output, fd_input, tape, g)
            t_output = [grad_output * torch.tensor(t).to(device) for i, t in enumerate(vjp_out)]

        return tuple(t_output)

    bases = (torch.autograd.Function,)
    members = {"forward": staticmethod(forward),
               "backward": staticmethod(backward)}

    return type(classname, bases, members)

    
def torchfireExit(comm) -> None:
    """
        Sends an exit signal to all workers on comm. 
        Call this function from the root process to exit torchfireRunWorker.
    """
    mpi_size = comm.Get_size()
    available_procs = [i for i in range(1, mpi_size)]
    for proc in available_procs:
        req = comm.isend([], proc, FLAGS.EXIT)
        req.wait()

    
def torchfireRunWorker(factory_functions: dict, built_functions: dict,
                       ensemble_comm, local_comm) -> None:
    """
        Torchfire worker processes sit in here waiting for work to do. 
        In this initial implementation, there are two things they can do:
        call factory functions provided by the factory_functions input dict, 
        and run functions that have been built. Arguments passed to the 
        factory functions are passed through a dictionary stored in data['data']
        where data is the data received from the root process. 
        Built functions are called by unpacking the arguments in data['data']. 

        Args: 
            factory_functions (dict): A dictionary mapping string names to factory functions.
                                      Factory functions are expected to return a tuple:
                                        (name, function_handle).
            built_functions (dict):   A dictionary mapping string names to functions that 
                                      are ready to be called for doing work. torchfireRunWorker
                                      has two call modes: CALL_FACTORY and NEW_DATA. 
                                      CALL_FACTORY looks for the function name in factory_functions
                                      and NEW_DATA looks for name in built_functions.
            ensemble_comm (MPI_Comm): MPI communicator to listen to for work. 
            local_comm (MPI_Comm):    MPI communicator used for local work. Automatically
                                      added to all factory function argument dictionaries 
                                      as args['local_comm'] = local_comm
                                      
    """
    while True:
        status = MPI.Status()
        data = ensemble_comm.recv(source=0, status=status)
        if status.tag == FLAGS.EXIT:
            break

        if status.tag == FLAGS.CALL_FACTORY:
            f = factory_functions[data['function_name']]
            args = data['data']
            args['local_comm'] = local_comm
            built_name, func_handle = f(args)
            built_functions[built_name] = func_handle
        elif status.tag == FLAGS.NEW_DATA:
            # first check to make sure the function exists
            # if data['function_name']...
            f = built_functions[data['function_name']]
            return_val = f(*data['data'])

            # assumes that return_val is a dictionary
            return_val['index'] = data['index']

            # mpi send results back to root
            req = ensemble_comm.isend(return_val, 0, FLAGS.RUN_FINISHED)
            req.wait()

    
def torchfireParallelMapReduce(function_name: str, comm, *data): # -> list:
    """
    Differentiable wrapper around MpiReduceTorchFunction 
    """
    f = MpiReduceTorchFunction()
    output = f.apply(function_name, comm, *data)
    output = output.sum()
    return output
    

def torchfireBuildFactory(function_name: str, args: dict, ensemble_comm) -> None:
    """
        Simple function wrapping the MPI calls to direct all 
        worker processes to build factory function with function_name. 
        Use individual send/recv since the worker processes are blocking 
        on a recv, not expecting a broadcast. This allows the worker 
        code to be more general and for us not to worry about synchonizing broadcast
        calls.        
    """
    mpi_size = ensemble_comm.Get_size()
    available_procs = [i for i in range(1, mpi_size)]
    compact_args = {
        'function_name': function_name,
        'data': args
    }
    for proc in available_procs:
        req = ensemble_comm.isend(compact_args, proc, FLAGS.CALL_FACTORY)
        req.wait()
