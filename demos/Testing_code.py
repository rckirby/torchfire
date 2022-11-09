import numpy as np
import firedrake
from fecr import from_numpy, to_numpy
import firedrake_adjoint  
import pyadjoint

from firedrake import (DirichletBC, FunctionSpace, TrialFunction, SpatialCoordinate, Constant, Function, sqrt, TestFunction, UnitSquareMesh, assemble, dx, grad, inner  )

mesh = UnitSquareMesh(5, 2)
V = FunctionSpace(mesh, "P", 1)
bc = DirichletBC(V, 0, (1, 2, 4))
x = SpatialCoordinate(mesh)


# Create tape associated with this forward pass
tape = pyadjoint.Tape()
pyadjoint.set_working_tape(tape)

u = Function(V)
v = TestFunction(V)
f = Constant(10.0)
A = (inner( grad(u), grad(v)) - inner( f , v ) ) * dx 
firedrake_output = assemble(A)
numpy_output = np.asarray(to_numpy(firedrake_output))


# Convert tangent covector (adjoint variable) to a backend variable
Δfiredrake_output = from_numpy(numpy_output, firedrake_output)

# pyadjoint doesn't allow setting Functions to block_variable.adj_value
Δfiredrake_output = Δfiredrake_output.vector()

tape.reset_variables()
firedrake_output.block_variable.adj_value = Δfiredrake_output

with tape.marked_nodes( (u,) ):
    tape.evaluate_adj(markings=True)
dfiredrake_inputs = (fi.block_variable.adj_value for fi in (u,) )

# Convert Firedrake gradients to NumPy array representation
dnumpy_inputs = tuple(
    None if dfi is None else np.asarray(to_numpy(dfi)) for dfi in dfiredrake_inputs
)

print(dnumpy_inputs)