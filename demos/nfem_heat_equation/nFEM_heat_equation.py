import firedrake
import numpy as np
import pandas as pd
import torch
from firedrake import (DirichletBC, FunctionSpace, Constant,
                       TestFunction, UnitSquareMesh, assemble, dx, grad, inner)
from path import Path
from torch import nn

from torchfire import fd_to_torch

device = torch.device('cpu')  # At the moment, we support CPU computation only.

# STEP 0. Initial parameters and training parameters
num_train_ultimate = 10000  # This ensures that larger data set provides more information for training
num_train = 600  # larger number means more data is given for training
num_test = 500

learning_rate = 1e-2
batch_size = 200
epochs = 1
neurons = 1000


# STEP 1. Loading data from .csv files
def load_data(name, target_shape=(-1,)):
    return torch.tensor(np.reshape(pd.read_csv(name).to_numpy(), target_shape)).to(device)


# 1.1 Loading train and test data
train_Observations_synthetic = load_data('data/Training_Solutions_u.csv', (num_train_ultimate, -1))
train_Observations_synthetic = train_Observations_synthetic[:num_train, :]

train_Parameters = load_data('data/Training_KL_Expansion_coefficients.csv', (num_train_ultimate, -1))
train_Parameters = train_Parameters[:num_train, :]

test_Observations_synthetic = load_data('data/Test_Solutions_u.csv', (num_test, -1))
test_Parameters = load_data('data/Test_KL_Expansion_coefficients.csv', (num_test, -1))

# 1.2 Loading eigenvalues, eigenvectors, observed indices, Degree of Freedom indices
nx, ny = 15, 15
dimension_of_PoI = (nx + 1) * (ny + 1)
num_truncated_series = 15

Eigen = load_data('data/Eigen_vector.csv', (dimension_of_PoI, num_truncated_series))
Sigma = load_data('data/Eigen_value_data.csv', (num_truncated_series, num_truncated_series))
free_index = load_data('data/Degree_of_Freedom_indices.csv')

# 1.3 Imposing boundary condition operator (trick)
Operator = np.zeros((210, 256))
i = 0
for j in free_index:
    Operator[i, j] = 1
    i += 1
Operator = torch.Tensor(Operator).to(device)

# 2. Physics Handler
mesh = UnitSquareMesh(15, 15)
V = FunctionSpace(mesh, "P", 1)
bc = DirichletBC(V, 0, (1, 2, 3))
templates = (firedrake.Function(V), firedrake.Function(V))


def assemble_firedrake(u, exp_kappa):
    v = TestFunction(u.function_space())
    f = Constant(20.0)
    return assemble(inner(exp_kappa * grad(u), grad(v)) * dx - inner(f, v) * dx, bcs=bc)


res_appply = fd_to_torch(assemble_firedrake, templates, "residualTorch").apply


# STEP 3. Building neural network
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

        # Mapping vectors z to nodal solutions at free nodes (excludes the boundary nodes)
        u = self.Neuralmap2(self.Relu(self.Neuralmap1(z.float())))

        # THIS IS IMPOSED BOUNDARY CONDITIONS
        u = torch.einsum('ij, bi -> bj', Operator, u)

        # generate kappa from vectors z
        kappa = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z.float())
        kappa = torch.exp(kappa)

        return u, kappa

    def ResidualTorch(self, u, kappa):
        """This generates the sum of residuals of all the samples within a batch given a batch of
        predicted solutions `u` and corresponding `exp(kappa)`

        Args:
            u (tensor): predicted solutions by neural network
            kappa (tensor): the corresponding kappa to predicted solutions

        Returns:
            scalar : the sum of residuals of all the samples within a batch
        """

        mse_loss = nn.MSELoss()
        residuals = torch.zeros(1, device=device)

        for u_nn_, kappa_ in zip(u, kappa):
            # Pass kappa and u through the torchfire-wrapped function to get a vector
            res_ = res_appply(u_nn_, kappa_)

            # Euclidean norm of that vector
            loss = mse_loss(res_, torch.zeros_like(res_))
            residuals = residuals + loss

        return residuals


model = NeuralNetwork().to(device)


# STEP 4. Training loss functions
def train_loop(model, optimizer, z):
    loss_train = 0
    for batch in range(int(num_train / batch_size)):
        u_train_pred, kappa = model(z[(batch) * batch_size:(batch + 1) * batch_size, :])

        # This computes the residual loss given the tensors exp(kapp) and the neural network that generates the solution u_nn
        residuals = model.ResidualTorch(u=u_train_pred, kappa=kappa)
        loss = residuals / batch_size
        loss_train += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_train


def test_loop(model, z_test, u_test_true):
    with torch.no_grad():
        u_test_pred, _ = model(z_test)
        test_u_acc = torch.mean(
            torch.linalg.vector_norm(u_test_pred - u_test_true, dim=-1) ** 2 / torch.linalg.vector_norm(u_test_true,
                                                                                                        dim=-1) ** 2)

    return test_u_acc


# STEP 5. Training process
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
functional = nn.MSELoss()


def numpy_formatter(np_array):
    return np.array2string(np_array, formatter={'float': lambda x: f'{x:.6f}'})


TRAIN_LOSS, TEST_ACC = [], []
for t in range(epochs):
    train_loss = train_loop(model, optimizer, train_Parameters)
    test_u_acc = test_loop(model, test_Parameters, test_Observations_synthetic)

    str_test_u_acc = numpy_formatter(test_u_acc.cpu().detach().numpy())
    str_train_loss = numpy_formatter(train_loss.cpu().detach().numpy()[0])

    print(f"Epoch {t + 1}\n-------------------------------")
    print(f"Test Acc:  {str_test_u_acc}  Train loss {str_train_loss} \n")

    # Save the training loss, testing accuracies and the neural network model for inference
    test_u_acc_old = 100
    if test_u_acc < test_u_acc_old:
        torch.save(model, 'results/best_model.pt')
        test_u_acc_old = test_u_acc

    TRAIN_LOSS.append(train_loss.cpu().detach().numpy())
    TEST_ACC.append(test_u_acc.cpu().detach().numpy())

# Step 6: Saving to the file
pd.DataFrame(np.asarray(TRAIN_LOSS)).to_csv(Path('results/train_loss.csv'), index=False)
pd.DataFrame(np.asarray(TEST_ACC)).to_csv(Path('results/test_acc.csv'), index=False)

print("Done!")
