import torch
import firedrake
import firedrake_adjoint  # noqa
import numpy as np
import ufl
from torchfire import fd_to_torch
import fdm
from fecr import evaluate_primal, evaluate_pullback

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
    F.interpolate( firedrake.sin(u * firedrake.pi) * firedrake.sin( x[1] * firedrake.pi))
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
inputs_2 = (torch.ones(V.dim(), requires_grad=True))
np_inputs_2 = np.array([t.detach().numpy().astype(np.float64()) for t in inputs_2])
ff_2 = lambda *args: evaluate_primal(firedrake_function, templates_2, *args)[0]



def test_torchfire_forward():
    numpy_output, _, _, _, = evaluate_primal(assemble_firedrake, templates, *np_inputs)
    bob = fd_to_torch(assemble_firedrake, templates, "bob")
    J = bob.apply(*inputs)

    assert np.isclose(numpy_output, J.detach().numpy())


def test_vjp_assemble_eval():
    numpy_output, _, _, _, = evaluate_primal(assemble_firedrake, templates, *np_inputs)

    bob = fd_to_torch(assemble_firedrake, templates, "bob")
    J = bob.apply(*inputs)
    vjp_out = torch.autograd.grad(J, inputs)

    fdm_jac0 = fdm.jacobian(ff0)(np_inputs[0])
    fdm_jac1 = fdm.jacobian(ff1)(np_inputs[1])
    fdm_jac2 = fdm.jacobian(ff2)(np_inputs[2])

    check1 = np.allclose(vjp_out[0], fdm_jac0)
    check2 = np.allclose(vjp_out[1], fdm_jac1)
    check3 = np.allclose(vjp_out[2], fdm_jac2)
    assert check1 and check2 and check3

def test_autograd_fecr_integration():
    numpy_output, _, _, _, = evaluate_primal(firedrake_function, templates_2, np_inputs_2)
    fdm_jac0 = 2 * np.ones_like(ff_2(np_inputs[0])) @ fdm.jacobian(ff_2)(np_inputs[0])

    bob = fd_to_torch(firedrake_function, templates_2, "bob")
    x = bob.apply
    y = torch.dot(x(inputs_2), x(inputs_2))
    norm_2_out = torch.autograd.grad(y, inputs_2, retain_graph=True, create_graph=True, allow_unused=True)

    assert np.allclose(norm_2_out[0].detach().numpy(), fdm_jac0)
