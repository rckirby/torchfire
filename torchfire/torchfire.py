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
        np_output, fd_output, fd_input, tape = evaluate_primal(
            fd_callable, templates, *inputs)
        ctx.save_for_backward(*inputs)
        ctx.stuff = (np_output, fd_output, fd_input, tape)

        return np_output

    def backward(ctx, grad_output):
        # inputs = ctx.saved_tensors
        np_output, fd_output, fd_input, tape = ctx.stuff
        g = np.ones_like(np_output)
        vjp_out = evaluate_pullback(fd_output, fd_input, tape, g)
        # todo: figure out shape of grad_output...

    bases = (torch.autograd.Function,)
    members = {"forward": staticmethod(forward),
               "backward": staticmethod(backward)}

    return type(classname, bases, members)


def assemble_firedrake(u, kappa0, kappa1):

    x = firedrake.SpatialCoordinate(mesh)
    f = x[0]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    J_form = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    J = firedrake.assemble(J_form)
    return J


mesh = firedrake.UnitSquareMesh(3, 2)
V = firedrake.FunctionSpace(mesh, "P", 1)
templates = (firedrake.Function(V),
             firedrake.Constant(0.0), firedrake.Constant(0.0))
inputs = (np.ones(V.dim()), np.ones(1) * 0.5, np.ones(1) * 0.6)

bob = fd_to_torch(assemble_firedrake, templates, "bob")

bob.forward(None, *inputs)
