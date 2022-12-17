import fdm
import firedrake
import firedrake_adjoint  # noqa
import numpy as np
import torch
import ufl
from fecr import evaluate_primal

from torchfire import fd_to_torch

mesh = firedrake.UnitSquareMesh(3, 2)
V = firedrake.FunctionSpace(mesh, "P", 1)


def assemble_firedrake(u, kappa0, kappa1):
    x = firedrake.SpatialCoordinate(mesh)
    f = x[0]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    J_form = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    J = firedrake.assemble(J_form)
    return J


def firedrake_function(u):
    x = firedrake.SpatialCoordinate(mesh)
    F = firedrake.Function(V)
    F.interpolate(firedrake.sin(u * firedrake.pi) * firedrake.sin(x[1] * firedrake.pi))
    return F


templates = (firedrake.Function(V), firedrake.Constant(0.0), firedrake.Constant(0.0))

inputs = (torch.ones(V.dim(), requires_grad=True), torch.ones(1, requires_grad=True) * 0.5,
          torch.ones(1, requires_grad=True) * 0.6)

np_inputs = tuple([t.detach().numpy().astype(np.float64()) for t in inputs])

ff = lambda *args: evaluate_primal(assemble_firedrake, templates, *args)[0]
ff0 = lambda x: ff(x, np_inputs[1], np_inputs[2])  # noqa: E731
ff1 = lambda y: ff(np_inputs[0], y, np_inputs[2])  # noqa: E731
ff2 = lambda z: ff(np_inputs[0], np_inputs[1], z)  # noqa: E731

templates_2 = (firedrake.Function(V),)
inputs_2 = (torch.ones(V.dim(), requires_grad=True),)

ff_2 = lambda *args: evaluate_primal(firedrake_function, templates_2, *args)[0]
ff_2_2 = lambda x: np.dot(ff_2(x), ff_2(x))  # noqa: E731


def test_torchfire_forward():
    numpy_output, _, _, _, = evaluate_primal(assemble_firedrake, templates, *np_inputs)
    bob = fd_to_torch(assemble_firedrake, templates, "bob")
    J = bob.apply(*inputs)

    assert np.isclose(numpy_output, J.detach().numpy())


def test_vjp_assemble_eval():
    bob = fd_to_torch(assemble_firedrake, templates, "bob")
    J = bob.apply(*inputs)
    grads = torch.ones_like(J)
    vjp_out = torch.autograd.grad(J, inputs, grad_outputs=grads)

    fdm_jac0 = fdm.jacobian(ff0)(np_inputs[0])
    fdm_jac1 = fdm.jacobian(ff1)(np_inputs[1])
    fdm_jac2 = fdm.jacobian(ff2)(np_inputs[2])

    check1 = np.allclose(vjp_out[0], fdm_jac0)
    check2 = np.allclose(vjp_out[1], fdm_jac1)
    check3 = np.allclose(vjp_out[2], fdm_jac2)
    assert check1 and check2 and check3


def test_autograd_fecr_integration():
    bob = fd_to_torch(firedrake_function, templates_2, "bob")
    x = bob.apply
    xx = x(*inputs_2)
    yy = xx @ xx
    grads = torch.ones_like(yy)

    grad_x, = torch.autograd.grad(yy, inputs_2, grad_outputs=grads, create_graph=True)
    fdm_jac0 = 2 * ff_2(np_inputs[0]) @ fdm.jacobian(ff_2)(np_inputs[0])
    assert np.allclose(grad_x.detach().numpy(), fdm_jac0)
