import firedrake
import numpy as np
import pandas as pd
import torch
from firedrake import (DirichletBC, FunctionSpace, Constant,
                       TestFunction, UnitSquareMesh, Function, solve, dx, grad, inner)
from torch import nn

from torchfire import fd_to_torch

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cpu")

# 0. Initial parameters and training parameters
num_train_ultimate = 10000
num_train = 100
num_test = 500
repeat_fac = 1  # Keep it 1 for now!

learning_rate = 1e-3
batch_size = num_train
epochs = 1
neurons = 5000

alpha = 8e3  # this value is the best for noise level of 0.005
noise_level = 0.005


# STEP 1. Loading data from .csv files
# Data for training and testing has been pre-built and exported to .csv files
# to avoid the need to generate data again each time the code is run.
# To Generate data, we follow two steps:
# 1. Drawing random kappa samples (train/test parameters) using KL expansion formula.
# 2. Solving Firedrake solver for solution state.
# 3. Applying the observation operator to achieve 10 observables

def load_data(name, target_shape=(-1,)):
    return torch.tensor(np.reshape(pd.read_csv(name).to_numpy(), target_shape)).to(device)


# 1.1 Loading train and test data
train_Observations_synthetic = load_data('data/Training_Sparse_Solutions_u.csv', (num_train_ultimate, -1))
train_Observations_synthetic = train_Observations_synthetic[:num_train, :].repeat(repeat_fac, 1)

train_Parameters = load_data('data/Training_KL_Expansion_coefficients.csv', (num_train_ultimate, -1))
train_Parameters = train_Parameters[:num_train, :].repeat(repeat_fac, 1)

test_Observations_synthetic = load_data('data/Test_Sparse_Solutions_u.csv', (num_test, -1))
test_Parameters = load_data('data/Test_KL_Expansion_coefficients.csv', (num_test, -1))

# Data randomization
train_Observations = train_Observations_synthetic + torch.normal(0, 1, train_Observations_synthetic.shape).to(
    device) * torch.unsqueeze(torch.max(train_Observations_synthetic, axis=1)[0].to(device), dim=1) * noise_level
test_Observations = test_Observations_synthetic + torch.normal(0, 1, test_Observations_synthetic.shape).to(
    device) * torch.unsqueeze(torch.max(test_Observations_synthetic, axis=1)[0].to(device), dim=1) * noise_level

# 1.2 Loading eigenvalues, eigenvectors, observed indices, Degree of Freedom indices, and sparse observables
nx, ny = 15, 15
num_observation = 10  # number of observed points
dimension_of_PoI = (nx + 1) * (ny + 1)
num_truncated_series = 15  # dimension of z vector

Eigen = load_data('data/Eigen_vector.csv', (dimension_of_PoI, num_truncated_series))
obs_indices = load_data('data/Observation_indices.csv', (num_observation, -1))
Sigma = load_data('data/Eigen_value_data.csv', (num_truncated_series, num_truncated_series))

# 2. Physics Handler
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, "P", 1)
bc = DirichletBC(V, 0, (1, 2, 3))
templates = (firedrake.Function(V), firedrake.Function(V))


def solve_firedrake(exp_u):
    y = Function(exp_u.function_space())
    v = TestFunction(exp_u.function_space())
    f = Constant(20.0)
    F = inner(exp_u * grad(y), grad(v)) * dx - f * v * dx
    solve(F == 0, y, bcs=bc)
    return y


# assemble_firedrake just takes a pair of functions now
templates = (firedrake.Function(V),)
diff_solver = fd_to_torch(solve_firedrake, templates, "solverTorch").apply


# STEP 3. Building neural network
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

    def forward(self, train_observations):
        """Forward pass before using Firedrake

        Args:
            train_observations (tensor): Sparse Observations

        Returns:
            z (tensor): predicted parameter z vector by neural network
        """

        # Mapping vectors u_obs to parameters z
        z_pred = self.Neuralmap2(self.Relu(self.Neuralmap1(train_observations.float())))

        # generate kappa from vectors z
        kappa = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z_pred.float())
        kappa = torch.exp(kappa)

        return z_pred, kappa

    def SolverTorch(self, kappa, train_observations):
        """This generates the solution for all the samples within a batch given a batch of
        predicted solutions `exp(u)`

        Args:
            kappa (tensor): predicted parameters obtained by neural network
            train_observations (tensor): Sparse Observations

        Returns:
             scalar : mean square error between reproduced observations and sparse true observations
        """

        loss_mc = torch.zeros(1, device=device)

        for _, (u_obs_true_, kappa_) in enumerate(zip(train_observations, kappa)):
            u_ = diff_solver(kappa_)[obs_indices].squeeze()

            loss_mc += torch.mean(torch.square(u_ - u_obs_true_)) * num_observation / num_train

        return loss_mc


model = NeuralNetwork().to(device)


# STEP 4. Training loss functions
def train_loop(model, optimizer, z, u_train_true, alpha):
    u_obs_batch, z_batch = u_train_true.float(), z.float()

    z_pred, kappa = model(u_obs_batch)
    loss_ml = torch.mean(torch.square(z_pred - z_batch * 0)) * num_truncated_series

    loss_mc = model.SolverTorch(kappa, u_obs_batch)
    loss = loss_ml + alpha * loss_mc

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def test_loop(model, z_test, u_test_true):
    with torch.no_grad():
        z_test_pred, _ = model(u_test_true)
        kappa_pred = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z_test_pred)
        kappa_true = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z_test.float())

    return torch.mean(
        torch.linalg.vector_norm(kappa_pred - kappa_true, dim=-1) ** 2 / torch.linalg.vector_norm(kappa_true,
                                                                                                  dim=-1) ** 2)


# STEP 5. Training process
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def numpy_formatter(np_array):
    return np.array2string(np_array, formatter={'float': lambda x: f'{x:.6f}'})


TRAIN_LOSS, TEST_ACC = [], []
for t in range(epochs):

    train_loss = train_loop(model, optimizer, train_Parameters, train_Observations, alpha)
    test_u_acc = test_loop(model, test_Parameters, test_Observations)

    str_test_u_acc = numpy_formatter(test_u_acc.cpu().detach().numpy())
    str_train_loss = numpy_formatter(train_loss.cpu().detach().numpy()[0])

    if t % 50 == 0:
        print(f"Epoch {t + 1}\n-------------------------------")
        print(f"Test Acc:  {str_test_u_acc} Train loss {str_train_loss} \n")

    # Save the training loss, testing accuracies and the neural network model for inference
    test_u_acc_old = 100
    if test_u_acc < test_u_acc_old:
        torch.save(model, 'results/best_model.pt')
        test_u_acc_old = test_u_acc

    TRAIN_LOSS.append(train_loss.cpu().detach().numpy())
    TEST_ACC.append(test_u_acc.cpu().detach().numpy())

# STEP 6: Saving to the file
pd.DataFrame(np.asarray(TRAIN_LOSS)).to_csv('results/train_loss.csv', index=False)
pd.DataFrame(np.asarray(TEST_ACC)).to_csv('results/test_acc.csv', index=False)

print("Done!")
