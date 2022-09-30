import torch
import firedrake
import firedrake_adjoint  # noqa
import numpy as np
import ufl

from fecr import evaluate_primal, evaluate_pullback


def fd_to_torch(fd_callable, templates, classname):
    """Creates a subclass of torch.autograd.Function implementing
    the static forward and backward methods in terms of fecr."""

    def forward(ctx, *inputs):
        np_inputs = [t.detach().numpy().astype(np.float64()) for t in inputs]
        np_output, fd_output, fd_input, tape = evaluate_primal(
            fd_callable, templates, *np_inputs)
        ctx.save_for_backward(*inputs)
        ctx.stuff = (np_output, fd_output, fd_input, tape)
        return torch.tensor(np_output, requires_grad=True)

    def backward(ctx, grad_output):
        #inputs, = ctx.saved_tensors
        np_output, fd_output, fd_input, tape = ctx.stuff
        g = np.ones_like(np_output)
        vjp_out = evaluate_pullback(fd_output, fd_input, tape, g)

        t_output = [ grad_output * torch.tensor(t, requires_grad=True) for t in vjp_out]
        grad_output = tuple(t_output)
        return grad_output

    bases = (torch.autograd.Function,)
    members = {"forward": staticmethod(forward),
               "backward": staticmethod(backward)}

    return type(classname, bases, members)




