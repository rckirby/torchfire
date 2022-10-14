import firedrake
# We need to import firedrake_adjoint to fire up the taping!
import firedrake_adjoint  # noqa
from fecr import evaluate_primal
from firedrake import (DirichletBC, FunctionSpace, SpatialCoordinate,
                       TestFunction, UnitSquareMesh, assemble, dx, grad, inner)
from torch.autograd import Variable
from torchfire import fd_to_torch

mesh = UnitSquareMesh(3, 2)
V = FunctionSpace(mesh, "P", 1)

# zero on 
bc = DirichletBC(V, 0, (1, 2, 4))


# This assumes that e^kappa will be computed outside of firedrake
# and stuffed into a piecewise linear FD function.
def assemble_firedrake(u, expkappa):
    x = SpatialCoordinate(mesh)
    v = TestFunction(u.function_space())
    f = x[0]

    return assemble(inner(expkappa * grad(u), grad(v)) * dx - inner(f, v) * dx, bcs=bc)


# assemble_firedrake just takes a pair of functions now
templates = (firedrake.Function(V), firedrake.Function(V))

# TODO:
# Write a function that creates a torch.Tensor
# tabulating the KL expansion, calls assemble_firedrake
# and takes the Euclidean norm of the vector.

# Longer-term performance TODO:
# `assemble` does a lot of Python futzing and bangs on a cache of
# precompiled variational forms before actually invoking the low-level code
# of calling it for small problems will probably swamp the cost of
# doing arithmetic.
#
# Firedrake has a `FormAssembler` that bypasses this.  It shouldn't be
# too hard to have our code call that instead.
# But we need to make sure it still works with the adjoining/differentiation.

# For now `assemble` fulfills the right contract, so we should run with that
# until we're ready to improve performance.
