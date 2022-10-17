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

# zero on 3 sides
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

# Need a function (Van & Jon?) that takes p (order of KL expansion) and a mesh
# (get coordinates from mesh.coordinates.dat.data)
# and returns a matrix A tabulating the KL eigenfunctions scaled by sqrt(lam_i)
# at each grid point.
#
# Then realizing kappa from the z vector is a matrix-vector product.
#
# Our loss function (Jorge!) takes random z and NN prediction u (both torch.tensors) and 
# 1.) compute kappa at mesh points via A @ z and
# 2.) exponentiate pointwise to get some ekappa tensory
# 3.) pass ekappa and u through the torchfire-wrapped function to get a vector
# 4.) return Euclidean norm of that vector.

# Afterwards, we can check that this really works and try to train a neural net with it!
