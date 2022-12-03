import numpy as np
import firedrake as fd
from fecr import from_numpy, to_numpy
from firedrake import (DirichletBC, FunctionSpace, SpatialCoordinate, Constant, Function,TestFunction, UnitSquareMesh, solve, dx, grad, inner)

from firedrake import (DirichletBC, FunctionSpace, SpatialCoordinate, Constant,
                       TestFunction, UnitSquareMesh, assemble, dx, grad, inner)
from torchfire import fd_to_torch
import firedrake
import torch
from torch import nn
device = torch.device('cpu')

from firedrake import *
import numpy as np
import numpy.linalg as la
import firedrake.mesh as fd_mesh
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from pathlib import Path

#! 0 CREATE THE TRANSFORM FROM FENICS TO FIREDRAKE, SAVE TO "Fenics_to_Firedrake"
n = 15 + 1
def create_column(row):
    colum = row
    vector = [colum]
    for i in range(row, n+row-1):
        colum += i
        vector.append(colum)
    return vector

Column1 = create_column(1)

def create_row(row):
    colum = Column1[row]
    vector = [colum]
    for i in range(row+2, n+row+1):
        colum += i
        vector.append(colum)
    return vector

Index_mat = np.zeros((n,n))
for i in range(n):
    Index_mat[i,:] = create_row(i)

Left = Index_mat - 1
Right = np.flip(n**2 - Left - 1)
Final = Right

for i in range(n):
    for j in range(n):
        if i+j < n:
            Final[i, j] = Left[i,j]

Final_Firedrake = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        Final_Firedrake[j,i] = Final[j,i]
# Final_Firedrake = Final

for k in range(2,n):
    for i in range(n-1):
        for j in range(1,n):
            if  i+j==k:
                Final_Firedrake[k,0] = Final[0,k]
                Final_Firedrake[j-1,i+1] = Final[j,i]
            
        
A = Final.astype(int).flatten()
B = Final_Firedrake.astype(int).flatten()

Transform = np.zeros((n**2,n**2))

for i in range((n**2)):
    Transform[B[i], A[i]] = 1

def fenics_to_Firedrake(u):
    return Transform @ u
def Firedrake_to_Fenics(u):
    return Transform.T @ u

pd.DataFrame(Transform.flatten()).to_csv(Path('data/Fenics_to_Firedrake.csv'), index=False)















# ? PLEASE TAKE A LOOK FROM HERE JODGE








#! 1. Loading data by pandas
num_train2 = 10000
num_train = 600
num_test = 500
repeat_fac = 1  # Keep it 1 for now!

train_input_file_name = 'poisson_2D_state_full_train_d' + str(num_train2) + '_n15_AC_1_1_pt5'
train_output_file_name = 'poisson_2D_parameter_train_d' + str(num_train2) + '_n15_AC_1_1_pt5'
test_input_file_name = 'poisson_2D_state_full_test_d' + str(num_test) + '_n15_AC_1_1_pt5'
test_output_file_name = 'poisson_2D_parameter_test_d' + str(num_test) + '_n15_AC_1_1_pt5'

df_train_Observations = pd.read_csv('data/' + train_input_file_name + '.csv')
df_train_Parameters = pd.read_csv('data/' + train_output_file_name + '.csv')
df_test_Observations = pd.read_csv('data/' + test_input_file_name + '.csv')
df_test_Parameters = pd.read_csv('data/' + test_output_file_name + '.csv')

train_Observations_synthetic = np.reshape(df_train_Observations.to_numpy(), (num_train2, -1))
train_Observations_synthetic = train_Observations_synthetic[:num_train, :]

train_Parameters = np.reshape(df_train_Parameters.to_numpy(), (num_train2, -1))
train_Parameters = train_Parameters[:num_train, :]

train_Observations_synthetic = (np.repeat(train_Observations_synthetic, repeat_fac, axis=0))
train_Parameters = (np.repeat(train_Parameters, repeat_fac, axis=0))

test_Observations_synthetic = (np.reshape(df_test_Observations.to_numpy(), (num_test, -1)))
test_Parameters = (np.reshape(df_test_Parameters.to_numpy(), (num_test, -1)))

print(train_Observations_synthetic.shape)
print(train_Parameters.shape)

print(test_Observations_synthetic.shape)
print(test_Parameters.shape)

n = 15
num_observation = 10  # number of observed points
dimension_of_PoI = (n + 1) ** 2  # external force field
num_truncated_series = 15

df_Eigen = pd.read_csv('data/Eigenvector_data' + '.csv')
df_Sigma = pd.read_csv('data/Eigen_value_data' + '.csv')

Eigen = (np.reshape(df_Eigen.to_numpy(), (dimension_of_PoI, num_truncated_series)))
Sigma = (np.reshape(df_Sigma.to_numpy(), (num_truncated_series, num_truncated_series)))

df_obs = pd.read_csv('data/poisson_2D_obs_indices_o10_n15' + '.csv')
obs_indices =  df_obs.to_numpy()

df_free_index = pd.read_csv('data/Free_index_data' + '.csv')
free_index = (df_free_index.to_numpy())

# ? generate kappa from vectors z
train_Parameters = np.einsum('ij,bj -> bi', np.matmul(Eigen, Sigma), train_Parameters)
train_Parameters = np.exp(train_Parameters)


# PCIK sample 
sample = 100

mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "P", 1)
bc = DirichletBC(V, 0, (1, 2, 3))
u = Function(V)
function = Function(V)

# Replace kappa with yours here
def solver_function(KAPPA, sample):
    kappa = KAPPA[sample,:]
    kappa_Firedrake = fenics_to_Firedrake(kappa)

    fd_exkappa = from_numpy(kappa_Firedrake, fd.Function(V)) # we need to explicitly provide template function for conversion

    v = TestFunction(u.function_space())
    f = Constant(20.0)
    F = inner(fd_exkappa * grad(u), grad(v)) * dx - inner(f, v) * dx
    solve(F == 0, u, bcs=bc)

    solutions = Firedrake_to_Fenics(to_numpy(u))
    return solutions


solutions = solver_function(train_Parameters, sample)


# ? THE SOLUTION AND KAPPA SHOULD GIVE ZEROS RESIDUALS!!! BUT IT DOES NOT???

# ! 1.5 TorchFire Mesh definE
templates = (firedrake.Function(V), firedrake.Function(V))

def assemble_firedrake(u, expkappa):
    x = SpatialCoordinate(mesh)
    v = TestFunction(u.function_space())
    f = Constant(20.0)

    return assemble(inner(expkappa * grad(u), grad(v)) * dx - inner(f, v) * dx, bcs=bc)

res = fd_to_torch(assemble_firedrake, templates, "residualTorch")
res_appply = res.apply

kappa_ = torch.tensor(fenics_to_Firedrake(train_Parameters[sample,:])).to(device)
un_ = torch.tensor(solutions).to(device)
res_ = res_appply(un_, kappa_)
print(res_)
print(torch.nn.MSELoss()(res_, torch.zeros_like(res_)))



















# # ! Plotting data to verify solvers
# fig = plt.figure(figsize=(10, 10))
# tricontourf(from_numpy(fenics_to_Firedrake(solutions), fd.Function(V)))
# plt.savefig("Firedrake_solution.png", dpi=150)
# plt.close()

# v1 = train_Observations_synthetic[sample,:]

# plot saving figure    
# fig = plt.figure(figsize=(10, 10))
# tricontourf(from_numpy(fenics_to_Firedrake(v1), fd.Function(V)))
# plt.savefig("Fenics_solution.png", dpi=150)
# plt.close()

# print('Fenics Firedrake relative error test: ', np.linalg.norm(v1 - solutions) /np.linalg.norm(solutions) )

# # ! Regenerating data from Firedrake solvers
# # ? TRAINING DATA
# train_Parameters = np.reshape(df_train_Parameters.to_numpy(), (num_train2, -1))
# train_Parameters = np.einsum('ij,bj -> bi', np.matmul(Eigen, Sigma), train_Parameters)
# train_Parameters = np.exp(train_Parameters)

# Obs = np.zeros((train_Parameters.shape[0], 256))
# Obs2 = np.zeros((train_Parameters.shape[0], 10))

# for i in range(train_Parameters.shape[0]):
#     sol = solver_function(train_Parameters, i)
#     Obs[i,:] = (sol).flatten()
#     Obs2[i,:] = (sol)[obs_indices].flatten()
    
# df_prior_mean = pd.DataFrame({'state': Obs.flatten()})
# df_prior_mean.to_csv('data/poisson_2D_state_full_train_d10000_n15_AC_1_1_pt5.csv', index=False)

# df_prior_mean = pd.DataFrame({'state_obs': Obs2.flatten()})
# df_prior_mean.to_csv('data/poisson_2D_state_obs_train_o10_d10000_n15_AC_1_1_pt5.csv', index=False)


# # ? TESTING DATA
# test_Parameters = np.einsum('ij,bj -> bi', np.matmul(Eigen, Sigma), test_Parameters)
# test_Parameters = np.exp(test_Parameters)

# Obs = np.zeros((test_Parameters.shape[0], 256))
# Obs2 = np.zeros((test_Parameters.shape[0], 10))

# for i in range(test_Parameters.shape[0]):
#     sol = solver_function(test_Parameters, i)
#     Obs[i,:] = (sol).flatten()
#     Obs2[i,:] = (sol)[obs_indices].flatten()

# df_prior_mean = pd.DataFrame({'state': Obs.flatten()})
# df_prior_mean.to_csv('data/poisson_2D_state_full_test_d500_n15_AC_1_1_pt5.csv', index=False)

# df_prior_mean = pd.DataFrame({'state_obs': Obs2.flatten()})
# df_prior_mean.to_csv('data/poisson_2D_state_obs_test_o10_d500_n15_AC_1_1_pt5.csv', index=False)


