import wandb
from torchfire import fd_to_torch
from scipy import sparse
from torch import nn
from firedrake import (DirichletBC, FunctionSpace, SpatialCoordinate, Constant,
                       TestFunction, UnitSquareMesh, Function, solve, dx, grad, inner)
import firedrake
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import os
os.environ["OMP_NUM_THREADS"] = "1"


torch.manual_seed(0)
np.random.seed(0)

# device = torch.device('cuda')
device = torch.device("cpu")

alpha = 8e3

# ! 0. Initial parameters and training parameters
num_train2 = 10000
num_train = 100
num_test = 500
repeat_fac = 1  # Keep it 1 for now!
learning_rate = 1e-3
batch_size = num_train
epochs = 15000
neurons = 5000

noise_level = 0.000

# ! 0.1 Using Wandb to upload the approach
filename = 'INVERSE_TNET_#train_' + str(num_train) + '_to_' + str(num_train * repeat_fac) + '_LR_' + str(int(learning_rate)) + '_batch_' + str(batch_size) + '_neurons_' + str(neurons)
wandb.init(project="Torch_Fire", entity="hainguyenpho", name=filename)
wandb.config.problem = 'Heat_TorchFire'
wandb.config.batchsize = batch_size
wandb.config.learning_rate = learning_rate
wandb.config.database = num_train


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

            # TODO: Modify this?
            loss_mc += torch.mean(torch.square(u_ - u_obs_true_)) * num_observation / num_train

        return loss_mc


# ! 3. Training functions
model = NeuralNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_loop(model, optimizer, z, u_train_true, load_f, alpha):
    u_obs_batch, z_batch = u_train_true.float(), z.float()

    z_pred, kappa = model(u_obs_batch)
    loss_ml = torch.mean(torch.square(z_pred - z_batch * 0)) * num_truncated_series

    loss_mc = model.SolverTorch(kappa, u_obs_batch)
    loss = loss_ml + alpha * loss_mc

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    z_pred, _ = model(u_train_true.float())
    kappa_pred = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z_pred)
    kappa_true = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z.float())
    train_u_acc = torch.mean(torch.linalg.vector_norm(kappa_pred - kappa_true, dim=-1) ** 2 / torch.linalg.vector_norm(kappa_true, dim=-1)**2)

    return loss, train_u_acc


def test_loop(model, z_test, u_test_true):

    with torch.no_grad():
        z_test_pred, _ = model(u_test_true)
        kappa_pred = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z_test_pred)
        kappa_true = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z_test.float())

    return torch.mean(torch.linalg.vector_norm(kappa_pred - kappa_true, dim=-1)**2 / torch.linalg.vector_norm(kappa_true, dim=-1)**2)


# ! 3. Training process# ! 3. Training process
TRAIN_LOSS, TEST_ACC = [], []

for t in range(epochs):

    train_loss, train_u_acc = train_loop(model, optimizer, train_Parameters, train_Observations, load_f, alpha)
    test_u_acc = test_loop(model, test_Parameters, test_Observations)

    wandb.log({"Test ACC": float(test_u_acc), "Train ACC": float(train_u_acc), "Train loss": float(train_loss)})

    str_test_u_acc = numpy_formatter(test_u_acc.cpu().detach().numpy())
    str_train_u_acc = numpy_formatter(train_u_acc.cpu().detach().numpy())
    str_train_loss = numpy_formatter(train_loss.cpu().detach().numpy()[0])
    if t % 50 == 0:
        print(f"Epoch {t + 1}\n-------------------------------")
        print(f"Test Acc:  {str_test_u_acc} Train Acc: {str_train_u_acc}  Train loss {str_train_loss} \n")

    # Save
    test_u_acc_old = 100
    if test_u_acc < test_u_acc_old:
        torch.save(model, 'best_model_INV.pt')
        test_u_acc_old = test_u_acc

    TRAIN_LOSS.append(train_loss.cpu().detach().numpy())
    TEST_ACC.append(test_u_acc.cpu().detach().numpy())

pd.DataFrame(np.asarray(TRAIN_LOSS)).to_csv(Path('data/TRAIN_LOSS.csv'), index=False)
pd.DataFrame(np.asarray(TEST_ACC)).to_csv(Path('data/TEST_ACC.csv'), index=False)

print("Done!")
