import os
os.environ["PATH"] += os.pathsep + '/home/georx/torchfire/firedrake/lib/python3.10/site-packages/graphviz'

import torch
import torch.nn as nn
import firedrake as fd
import firedrake_adjoint  # noqa
import numpy as np
import ufl
from torchfire import fd_to_torch
import fdm
from fecr import evaluate_primal, evaluate_pullback
from torch.autograd import Variable
import torchviz
import matplotlib.pyplot as plt

#def neural_net(n_input, n_hidden, n_output):
#    model = nn.Sequential(nn.Flatten(),
#                          nn.Linear(n_input, n_hidden),
#                          nn.LeakyReLU(),
#                          nn.Linear(n_hidden, n_output),
#                          nn.ReLU())
#    return model

class neural_net(torch.nn.Module):

    def __init__(self):
        super(neural_net, self).__init__()
        self.linear1 = nn.Linear(27, 32)
        self.activation1 = nn.ReLU()
        self.linear2 = torch.nn.Linear(32, 32)
        self.activation2 = nn.ReLU()
        self.linear3 = torch.nn.Linear(32, 32)
        self.activation3 = nn.ReLU()
        self.linear4 = torch.nn.Linear(32, 27)
        self.activation4 = nn.Tanh()

    def forward(self, x):
        x = torch.flatten(x)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.activation4(x)
        return x

def a_function(f):
    V = fd.FunctionSpace(mesh, "P", 1)
    x = fd.SpatialCoordinate(mesh)

    u = fd.Function(V)
    v = fd.TestFunction(V)
    bcs = [fd.DirichletBC(V, fd.Constant(2.0), (1,))]

    a = (fd.inner(fd.grad(u), fd.grad(v))  - f * v  )* fd.dx

    fd.solve( a  == 0, u, bcs=bcs)
    return u

def F_function(f):
    x = fd.SpatialCoordinate(mesh)
    F = fd.Function(V)
    F.interpolate(fd.sin(x[0] * fd.pi) * fd.sin(2 * x[1] * fd.pi))
    return F




#############################

N = 8
mesh = fd.UnitSquareMesh(N, 2)
V = fd.FunctionSpace(mesh, "P", 1)
V_f = fd.FunctionSpace(mesh, "P", 1)
x = fd.SpatialCoordinate(mesh)
print(V.dim())
print(V_f.dim())
###########################

F_templates = (fd.Function(V), )
F_input = (torch.ones(V.dim(), requires_grad=False),)
F = fd_to_torch(F_function, F_templates, "bob_F")
F_ = F.apply(*F_input)

###########################

templates = (fd.Function(V_f),)
inputs = (torch.ones(V_f.dim(), requires_grad=True),)
a = fd_to_torch(a_function, templates, "bob_a")
a_ = a.apply

#############################

error_L2 = nn.MSELoss()
learning_rate = 0.0001
f_nn = neural_net() #neural_net(N, 68, N)
optimizer = torch.optim.Adam(f_nn.parameters(), lr=learning_rate)
losses = []

for epoch in range(3500):
    print(f"Epoch:{epoch}")
    f = f_nn(*inputs)
    y = a_(f)

    loss = error_L2(y, F_)
    grad_x, = torch.autograd.grad(loss, inputs, create_graph=True)
    torchviz.make_dot((grad_x, inputs[0] , loss), params={"grad_x": grad_x, "x": inputs[0], "out": loss}).render("attached",format="png")
    losses.append(loss)
    print(f"Loss:{loss}")

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot([loss.detach() for loss in losses][1:])
plt.savefig('losses.png')
plt.close()
print("Predicted f:", f.detach().numpy())
print("True f:", F_)
plt.plot(f.detach().numpy())
plt.savefig('f.png')
