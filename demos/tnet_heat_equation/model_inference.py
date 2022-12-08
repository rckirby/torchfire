import matplotlib
from firedrake import *
import firedrake as fd
import matplotlib.pyplot as plt
from fecr import from_numpy, to_numpy
import numpy as np
import math
import pandas as pd

import firedrake

import torch
from torch import nn
from scipy import sparse

# device = torch.device('cuda')
from torchfire import fd_to_torch

device = torch.device("cpu")

alpha = 8e3

# ! 0. Initial parameters and training parameters
num_train2 = 10000
num_train = 100
num_test = 500
repeat_fac = 1  # Keep it 1 for now!
learning_rate = 1e-3
batch_size = num_train
epochs = 50000
neurons = 5000

noise_level = 0.000

# # ! 0.1 Using Wandb to upload the approach
# filename = 'INVERSE_TNET_#train_' + str(num_train) + '_to_' + str(num_train * repeat_fac) + '_LR_' + str(int(learning_rate)) + '_batch_' + str(batch_size) + '_neurons_' + str(neurons)
# wandb.init(project="Torch_Fire", entity="hainguyenpho", name=filename)
# wandb.config.problem = 'Heat_TorchFire'
# wandb.config.batchsize = batch_size
# wandb.config.learning_rate = learning_rate
# wandb.config.database = num_train


# ! 1. Loading data by pandas
train_input_file_name = 'poisson_2D_state_obs_train_o10_d' + str(num_train2) + '_n15_AC_1_1_pt5'
train_output_file_name = 'poisson_2D_parameter_train_d' + str(num_train2) + '_n15_AC_1_1_pt5'
test_input_file_name = 'poisson_2D_state_obs_test_o10_d' + str(num_test) + '_n15_AC_1_1_pt5'
test_output_file_name = 'poisson_2D_parameter_test_d' + str(num_test) + '_n15_AC_1_1_pt5'

df_train_Observations = pd.read_csv('data/' + train_input_file_name + '.csv')
df_train_Parameters = pd.read_csv('data/' + train_output_file_name + '.csv')
df_test_Observations = pd.read_csv('data/' + test_input_file_name + '.csv')
df_test_Parameters = pd.read_csv('data/' + test_output_file_name + '.csv')

train_Observations_synthetic = np.reshape(df_train_Observations.to_numpy(), (num_train2, -1))
train_Observations_synthetic = train_Observations_synthetic[:num_train, :]

train_Parameters = np.reshape(df_train_Parameters.to_numpy(), (num_train2, -1))
train_Parameters = train_Parameters[:num_train, :]

train_Observations_synthetic = torch.tensor(np.repeat(train_Observations_synthetic, repeat_fac, axis=0)).to(device)
train_Parameters = torch.tensor(np.repeat(train_Parameters, repeat_fac, axis=0)).to(device)

test_Observations_synthetic = torch.tensor(np.reshape(df_test_Observations.to_numpy(), (num_test, -1))).to(device)
test_Parameters = torch.tensor(np.reshape(df_test_Parameters.to_numpy(), (num_test, -1))).to(device)

print(train_Observations_synthetic.shape)
print(train_Parameters.shape)

print(test_Observations_synthetic.shape)
print(test_Parameters.shape)

# ? 1.1 Add noise (WE DO NOT USE vmap, BE CAREFUL! )
train_Observations = train_Observations_synthetic + torch.normal(0, 1, train_Observations_synthetic.shape).to(
    device) * torch.unsqueeze(torch.max(train_Observations_synthetic, axis=1)[0].to(device), dim=1) * noise_level
test_Observations = test_Observations_synthetic + torch.normal(0, 1, test_Observations_synthetic.shape).to(
    device) * torch.unsqueeze(torch.max(test_Observations_synthetic, axis=1)[0].to(device), dim=1) * noise_level

# ! 1.2 Loading eigenvalues, eigenvectors
# ? 1.2 Load Eigenvalue, Eigenvectors, observed indices, prematrices
# ? Physical model information
n = 15
num_observation = 10  # number of observed points
dimension_of_PoI = (n + 1)**2  # number of grid points
num_truncated_series = 15  # dimension of z vector

df_Eigen = pd.read_csv('data/Eigenvector_data' + '.csv')
df_Sigma = pd.read_csv('data/Eigen_value_data' + '.csv')

Eigen = torch.tensor(np.reshape(df_Eigen.to_numpy(), (dimension_of_PoI, num_truncated_series))).to(device)
Sigma = torch.tensor(np.reshape(df_Sigma.to_numpy(), (num_truncated_series, num_truncated_series))).to(device)

df_obs = pd.read_csv('data/poisson_2D_obs_indices_o10_n15' + '.csv')
obs_indices = torch.tensor(np.reshape(df_obs.to_numpy(), (num_observation, -1))).to(device).squeeze()

df_free_index = pd.read_csv('data/Free_index_data' + '.csv')
free_index = torch.tensor(df_free_index.to_numpy()).to(device).squeeze()

boundary_matrix = sparse.load_npz('data/boundary_matrix_n15' + '.npz')
pre_mat_stiff_sparse = sparse.load_npz('data/prestiffness_n15' + '.npz')
load_vector_n15 = sparse.load_npz('data/load_vector_n15' + '.npz')
load_vector = sparse.csr_matrix.todense(load_vector_n15).T

Prematrix = torch.tensor(pre_mat_stiff_sparse.toarray()).to(device)
load_f = torch.tensor(load_vector).to(device)

Operator = np.zeros((210, 256))
i = 0
for j in free_index:
    Operator[i, j] = 1
    i += 1
Operator = torch.Tensor(Operator).to(device)

# ! 1.3 Firedrake and Fenics switch matrix
Fenics_to_Fridrake_mat = torch.tensor(np.reshape(pd.read_csv('data/Fenics_to_Firedrake' + '.csv').to_numpy(), ((n + 1)**2, (n + 1)**2))).to(device)


def Fenics_to_Fridrake(u):
    # Fenics_to_Fridrake_mat @ u
    return torch.einsum('ij, j -> i', Fenics_to_Fridrake_mat.float(), u.float())


def Fridrake_to_Fenics(u):
    # Fenics_to_Fridrake_mat.T @ u
    return torch.einsum('ij, i -> j', Fenics_to_Fridrake_mat.float(), u.float())


# ! 1.5 TorchFire Mesh definition
mesh = UnitSquareMesh(15, 15)
V = FunctionSpace(mesh, "P", 1)
bc = DirichletBC(V, 0, (1, 2, 3))
templates = (firedrake.Function(V), firedrake.Function(V))


def solve_firedrake(exp_u):
    x = SpatialCoordinate(mesh)
    y = Function(exp_u.function_space())
    v = TestFunction(exp_u.function_space())
    f = Constant(20.0)
    F = inner(exp_u * grad(y), grad(v)) * dx - f * v * dx
    solve(F == 0, y, bcs=bc)
    return y

# assemble_firedrake just takes a pair of functions now
templates = (firedrake.Function(V),)

diff_solver = fd_to_torch(solve_firedrake, templates, "solverTorch").apply


def numpy_formatter(np_array):
    """
    It takes a numpy array and returns a string representation of the array with 6 decimal places

    :param np_array: The array to be printed
    :return: the array as a string, with the float values rounded to 6 decimal places.
    """
    return np.array2string(np_array, formatter={'float': lambda x: f'{x:.6f}'})

# ! 2. Building neural network


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.Neuralmap1 = nn.Linear(num_observation, neurons)
        self.Relu = nn.ReLU()
        self.Neuralmap2 = nn.Linear(neurons, num_truncated_series)
        torch.nn.init.normal_(self.Neuralmap1.weight, mean=0.0, std=.02)
        torch.nn.init.normal_(self.Neuralmap1.bias, mean=0.0, std=.000)
        torch.nn.init.normal_(self.Neuralmap2.weight, mean=0.0, std=.02)
        torch.nn.init.normal_(self.Neuralmap2.bias, mean=0.0, std=.000)

    def forward(self, u):
        """Forward pass before using FireDrake

        Args:
            u (tensor): the train observable vector u_obs

        Returns:
            z (tensor): parameter z vector
        """

        # ? Mapping vectors u_obs to parameters z
        z_pred = self.Neuralmap2(self.Relu(self.Neuralmap1(u.float())))

        # ? generate kappa from vectors z
        kappa = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z_pred.float())
        kappa = torch.exp(kappa)

        return z_pred, kappa

    def SolverTorch(self, kappa, train_observations):
        """This generates the solution for all the samples within a batch given a batch of
        predicted solutions `exp(u)`

        Args:
            exp_u (tensor): predicted solutions of neural network

        Returns:
             torch.Tensor: the solutions for each pde in the batch
        """

        loss_mc = torch.zeros(1, device=device)

        for i, (u_obs_true_, kappa_) in enumerate(zip(train_observations, kappa)):

            kappa_ = Fenics_to_Fridrake(kappa_)
            u_ = Fridrake_to_Fenics(diff_solver(kappa_).squeeze())[obs_indices]

            loss_mc += torch.mean(torch.square(u_ - u_obs_true_)) * num_observation / num_train

        return loss_mc


# ! 3. Training functions
model = NeuralNetwork().to(device)

n = 15
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "P", 1)

# ! 1.3 Firedrake and Fenics switch matrix
Fenics_to_Fridrake_mat = torch.tensor(np.reshape(pd.read_csv('data/Fenics_to_Firedrake' + '.csv').to_numpy(), ((n + 1)**2, (n + 1)**2))).to(device)


def Fenics_to_Fridrake(u):
    # Fenics_to_Fridrake_mat @ u
    return torch.einsum('ij, bj -> bi', Fenics_to_Fridrake_mat.float(), u.float())


def Fridrake_to_Fenics(u):
    # Fenics_to_Fridrake_mat.T @ u
    return torch.einsum('ij, bi -> bj', Fenics_to_Fridrake_mat.float(), u.float())


def plot_u(u, u_pred, i):
    # plot saving figure
    plt.figure(figsize=(13, 6))
    max_u = math.ceil(max(np.max(u[i, :]), np.max(u_pred[i, :]))*10+1)/10
    min_u = math.floor(min(np.min(u[i, :]), np.min(u_pred[i, :]))*10-1)/10
    levels = np.arange(min_u,max_u,0.1)
    
    plt.subplot(121)
    ax = plt.gca()
    ax.set_aspect("equal")
    l = tricontourf(from_numpy(np.reshape(u[i, :], (256, 1)), fd.Function(V)), axes=ax, levels=levels)
    triplot(mesh, axes=ax, interior_kw=dict(alpha=0.05))
    plt.colorbar(l, fraction=0.046, pad=0.04)
    plt.title('True ' + str(i) + 'th conductivity field ' + r'$\kappa$')

    plt.subplot(122)
    ax = plt.gca()
    ax.set_aspect("equal")
    l = tricontourf(from_numpy(np.reshape(u_pred[i, :], (256, 1)), fd.Function(V)), axes=ax, levels=levels)
    triplot(mesh, axes=ax, interior_kw=dict(alpha=0.05))
    plt.colorbar(l, fraction=0.046, pad=0.04)
    plt.title('Predicted ' + str(i) + 'th conductivity field ' + r'$\kappa$ by TNet-TorchFire')

    plt.savefig("results/predicted_solutions/pred_" + str(i) + ".png", dpi=600, bbox_inches='tight')
    plt.close()


# Load
model_best = torch.load('results/best_model.pt')
z_pred, kappa_pred = model_best(test_Observations)
z_pred = z_pred
z_true = test_Parameters

kappa_pred = Fenics_to_Fridrake(torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z_pred.float()))
kappa_true = Fenics_to_Fridrake(torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z_true.float()))

kappa_pred = kappa_pred.cpu().detach().numpy().astype(np.float64)
kappa_true = kappa_true.cpu().detach().numpy().astype(np.float64)

# torch.mean(torch.linalg.vector_norm(torch.tensor(kappa_pred).to(device) - torch.tensor(kappa_true).to(device), dim=-1)**2 / torch.linalg.vector_norm(torch.tensor(kappa_true).to(device), dim=-1)**2)

Cases = 40
for sample in range(Cases):
    plot_u(kappa_true, kappa_pred, sample)

import imageio.v2
image_list = []
for step in range(Cases):
     image_list.append(imageio.v2.imread("results/predicted_solutions/pred_" + str(step) + ".png"))
imageio.mimwrite('results/animations.gif', image_list, duration=0.5)
# # imageio.mimwrite('animated_burger_sample_' + str(sample) + '.gif', image_list, fps = 60)

