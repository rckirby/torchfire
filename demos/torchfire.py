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
        device = inputs[0].device
        np_inputs = [t.cpu().detach().numpy().astype(np.float64()) for t in inputs]
        np_output, fd_output, fd_input, tape = evaluate_primal(
            fd_callable, templates, *np_inputs)
        ctx.save_for_backward(*inputs)
        ctx.stuff = (np_output, fd_output, fd_input, tape)
        output = torch.as_tensor(np_output, dtype=torch.float64, device=device)
        return output

    def backward(ctx, grad_output):
        # print(torch.norm(grad_output))
        inputs = ctx.saved_tensors
        device = inputs[0].device
        np_output, fd_output, fd_input, tape = ctx.stuff
        g = np.ones_like(np_output)
        vjp_out = evaluate_pullback(fd_output, fd_input, tape, g)
        
        if len( list(grad_output.size())) > 0:
            t_output = (torch.outer(grad_output, torch.from_numpy(t).to(device)) for i, t in enumerate(vjp_out))
        else:
            t_output = (grad_output * torch.from_numpy(t).to(device) for i, t in enumerate(vjp_out))
        
        return tuple(t_output)

    bases = (torch.autograd.Function,)
    members = {"forward": staticmethod(forward),
               "backward": staticmethod(backward)}

    return type(classname, bases, members)