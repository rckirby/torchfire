from mpi4py import MPI
import torch
from time import sleep

class FLAGS:
    RECEIVED = 1
    RUN_FINISHED = 2
    EXIT = 3
    NEW_DATA = 4
    CALL_FACTORY = 5

class MpiReduceTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function_name, comm, *data):
        """
        This function acts as the scheduler for calling function_name
        across all items in data. It should only be called my 
        the root of comm as it will dispatch work to worker processes 
        via send/recv calls and monitor processes to determine when the work is done. 

        Args: 
            function_name (str): This is the function name as a string. It will be looked 
                                 up on the worker processes registered_functions dict. 
                                 If not present, it will raise an exception. 
                                 Function should return a dictionary with keys 'val' and 'grad'.
            data (tensors):      This contains a list of the data that will be looped over. 
                                 function_name is called as:
                                   registered_functions[function_name](*(data.pop()))
                                 so data should be a list of tuples or a list of lists.
                                 This can easily be created for multiple lists 
                                 by zipping lists together: data = zip(list1, list2)
            comm (MPI comm):     This is the communicator over which the data will be 
                                 sent. This is the ensemble comm, not the local comm.
                                 
        Returns:
            List of results, the same size as data. Reduction is done on the calling code.
        """
        mpi_size = comm.Get_size()
        available_procs = [i for i in range(1, mpi_size)]
        solutions = []
        # this needs changed to be general. Hardcoding now to test if it works first.
        grads = torch.zeros_like(data[1], requires_grad=True)

        # initialize scenarios left to the total number of scenarios to run
        scenarios_left = len(data[0])
        data = list(zip(*data))
        curr_index = 0
        while scenarios_left > 0:
            # check workers for results
            s = MPI.Status()
            comm.Iprobe(status=s)
            
            if s.tag == FLAGS.RUN_FINISHED:
                outputs = comm.recv()
                solutions.append(outputs['val'])
                grads[outputs['index'], :] = outputs['grad']
                #grads.append(outputs['grad'])
                scenarios_left -= 1
                available_procs.append(s.source)

            # assign more work if there are free processes
            if len(available_procs) > 0 and len(data) > 0:
                curr_proc = available_procs.pop(0)
                curr_data = data.pop(0) # assumes data is list of list or list of tuple
                # need to add function name to dictionary
                compact_data = {
                    'function_name': function_name,
                    'data': curr_data,
                    'index': curr_index
                }
                curr_index += 1
                
                # block here to ensure the process starts before moving on so we don't overwrite buffer
                req = comm.isend(compact_data, curr_proc, FLAGS.NEW_DATA)
                req.wait()
            elif len(available_procs) == 0:
                # if there are no processes available, wait a bit
                sleep(0.1)
                
        reduced_sol = torch.zeros_like(solutions[0], requires_grad=True)
        for sol in solutions:
            reduced_sol += sol
        
        # we assume that the only gradients that matter are the gradients
        # of the output w.r.t. the input
        print(grads, flush=True)
        ctx.save_for_backward(None, grads)
        return reduced_sol
        
    @staticmethod
    def backward(ctx, grad_outputs):
        grads = ctx.saved_tensors
        return None, None, *grads
