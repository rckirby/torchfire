from firedrake import *
import firedrake as fd
import matplotlib.pyplot as plt
from fecr import from_numpy, to_numpy
import numpy as np
import pandas as pd

import torch
from torch import nn
from scipy import sparse

# device = torch.device('cuda')
from torchfire import fd_to_torch

device = torch.device("cpu")

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

# %%
# ! 0. Initial parameters and training parameters
num_train2 = 10000
num_train = 600
num_test = 500
repeat_fac = 1  # Keep it 1 for now!
# %%
learning_rate = 1e-2
batch_size = 200
epochs = 1000
neurons = 1000

# ! 1. Loading data by pandas
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

train_Observations_synthetic = torch.tensor(np.repeat(train_Observations_synthetic, repeat_fac, axis=0)).to(device)
train_Parameters = torch.tensor(np.repeat(train_Parameters, repeat_fac, axis=0)).to(device)

test_Observations_synthetic = torch.tensor(np.reshape(df_test_Observations.to_numpy(), (num_test, -1))).to(device)
test_Parameters = torch.tensor(np.reshape(df_test_Parameters.to_numpy(), (num_test, -1))).to(device)

print(train_Observations_synthetic.shape)
print(train_Parameters.shape)

print(test_Observations_synthetic.shape)
print(test_Parameters.shape)

# %% [markdown]
# ! 1.2 Loading eigenvalues, eigenvectors
# ? 1.2 Load Eigenvalue, Eigenvectors, observed indices, prematrices
# ? Physical model information
n = 15
num_observation = 10  # number of observed points
dimension_of_PoI = (n + 1)**2  # external force field
num_truncated_series = 15

df_Eigen = pd.read_csv('data/Eigenvector_data' + '.csv')
df_Sigma = pd.read_csv('data/Eigen_value_data' + '.csv')

Eigen = torch.tensor(np.reshape(df_Eigen.to_numpy(), (dimension_of_PoI, num_truncated_series))).to(device)
Sigma = torch.tensor(np.reshape(df_Sigma.to_numpy(), (num_truncated_series, num_truncated_series))).to(device)

df_obs = pd.read_csv('data/poisson_2D_obs_indices_o10_n15' + '.csv')
obs_indices = np.reshape(df_obs.to_numpy(), (num_observation, -1))

df_free_index = pd.read_csv('data/Free_index_data' + '.csv')
free_index = torch.tensor(df_free_index.to_numpy()).to(device)

boundary_matrix = sparse.load_npz('data/boundary_matrix_n15' + '.npz')
pre_mat_stiff_sparse = sparse.load_npz('data/prestiffness_n15' + '.npz')
load_vector_n15 = sparse.load_npz('data/load_vector_n15' + '.npz')
load_vector = sparse.csr_matrix.todense(load_vector_n15).T

# Acoo = pre_mat_stiff_sparse.tocoo()
# Prematrix = torch.sparse.LongTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
#                               torch.LongTensor(Acoo.data.astype(np.int32))).to(device)
Prematrix = torch.tensor(pre_mat_stiff_sparse.toarray()).to(device)
load_f = torch.tensor(load_vector).to(device)


Operator = np.zeros((210, 256))
i = 0
for j in free_index:
    Operator[i, j] = 1
    i += 1

Operator = torch.Tensor(Operator).to(device)


# ! 2. Building neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.Neuralmap1 = nn.Linear(15, neurons)
        self.Relu = nn.ReLU()
        self.Neuralmap2 = nn.Linear(neurons, 210)
        torch.nn.init.normal_(self.Neuralmap1.weight, mean=0.0, std=.01)
        torch.nn.init.normal_(self.Neuralmap2.weight, mean=0.0, std=.01)

    def forward(self, z):
        """Forward pass before using FireDrake

        Args:
            z (tensor): the train vectors z

        Returns:
            u (tensor): the predicted solutions from vectors z
            kappa (tensor): the kappa is transformed through eigenpairs
        """

        # ? Mapping vectors z to nodal solutions at free nodes (excludes the boundary nodes)
        u = self.Neuralmap2(self.Relu(self.Neuralmap1(z.float())))

        # ? THIS IS IMPOSED BOUNDARY CONDITIONS
        u = torch.einsum('ij, bi -> bj', Operator, u)
        u = Fenics_to_Fridrake(u)

        # ? generate kappa from vectors z
        kappa = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z.float())
        # kappa = torch.exp(kappa)
        kappa = Fenics_to_Fridrake(kappa)

        return u, kappa

    def ResidualTorch(self, u, kappa):
        """This generates the sum of residuals of all the samples within a batch given a batch of
        predicted solutions `u` and corresponding `exp(kappa)`

        Args:
            u (tensor): predicted solutions of neural network
            kappa (tensor): the corresponding kappa to predicted solutions

        Returns:
            scalar : the sum of residuals of all the samples within a batch
        """

        mse_loss = nn.MSELoss()
        res = fd_to_torch(assemble_firedrake, templates, "residualTorch")
        residuals = torch.zeros(1, device=device)
        res_appply = res.apply
        for u_nn_, kappa_ in zip(u, kappa):
            # Pass kappa and u through the torchfire-wrapped function to get a vector
            res_ = res_appply(u_nn_, kappa_)
            # Euclidean norm of that vector

            loss = mse_loss(res_, torch.zeros_like(res_))
            residuals = residuals + loss

        return residuals


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


def plot_u(u, u_pred, kappa, i):
    # plot saving figure
    plt.figure(figsize=(17, 6))

    plt.subplot(131)
    ax = plt.gca()
    ax.set_aspect("equal")
    l = tricontourf(from_numpy(np.reshape(kappa[i, :], (256, 1)), fd.Function(V)), axes=ax)
    triplot(mesh, axes=ax, interior_kw=dict(alpha=0.05))
    plt.colorbar(l, fraction=0.046, pad=0.04)
    plt.title(str(i) + 'th conductivity field ' + r'$\kappa$')

    plt.subplot(132)
    ax = plt.gca()
    ax.set_aspect("equal")
    l = tricontourf(from_numpy(np.reshape(u[i, :], (256, 1)), fd.Function(V)), axes=ax)
    triplot(mesh, axes=ax, interior_kw=dict(alpha=0.05))
    plt.colorbar(l, fraction=0.046, pad=0.04)
    plt.title('True ' + str(i) + 'th Test Solution by Firedrake')

    plt.subplot(133)
    ax = plt.gca()
    ax.set_aspect("equal")
    l = tricontourf(from_numpy(np.reshape(u_pred[i, :], (256, 1)), fd.Function(V)), axes=ax)
    triplot(mesh, axes=ax, interior_kw=dict(alpha=0.05))
    plt.colorbar(l, fraction=0.046, pad=0.04)
    plt.title('Predicted ' + str(i) + 'th Solution by nFEM')

    plt.savefig("results/predicted_solutions/pred_" + str(i) + ".png", dpi=600, bbox_inches='tight')
    plt.close()


True_u = Fenics_to_Fridrake(test_Observations_synthetic).cpu().detach().numpy().astype(np.float64)

# Load
model_best = torch.load('results/best_model.pt')
u_pred, kappa = model_best(test_Parameters)
u_pred = (u_pred).cpu().detach().numpy().astype(np.float64)
kappa = (kappa).cpu().detach().numpy().astype(np.float64)

Cases = 40
for sample in range(Cases):
    plot_u(True_u, u_pred, kappa, sample)


import imageio.v2
image_list = []
for step in range(Cases):
    image_list.append(imageio.v2.imread("results/predicted_solutions/pred_" + str(step) + ".png"))
imageio.mimwrite('results/animations.gif', image_list, duration=0.5)
# imageio.mimwrite('animated_burger_sample_' + str(sample) + '.gif', image_list, fps = 60)

