import math
import imageio.v2
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fecr import from_numpy
from torch import nn

device = torch.device('cpu')  # At the moment, we support CPU computation only.

# STEP 0. Initial parameters and training parameters
num_train_ultimate = 10000  # This ensures that larger data set provides more information for training
num_train = 600  # larger number means more data is given for training
num_test = 500

learning_rate = 1e-2
batch_size = 200
epochs = 1000
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

# 2. Physics mesh and function space for plotting function in Firedrake
mesh = fd.UnitSquareMesh(nx, ny)
V = fd.FunctionSpace(mesh, "P", 1)


# 3. Building neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.Neuralmap1 = nn.Linear(15, neurons)
        self.Relu = nn.ReLU()
        self.Neuralmap2 = nn.Linear(neurons, 210)
        torch.nn.init.normal_(self.Neuralmap1.weight, mean=0.0, std=.01)
        torch.nn.init.normal_(self.Neuralmap2.weight, mean=0.0, std=.01)

    def forward(self, z):
        """Forward pass before using Firedrake

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


model = NeuralNetwork().to(device)


# 4. Plotting function
def plot_u(u, u_pred, kappa, i):
    plt.figure(figsize=(17, 6))

    max_kappa = math.ceil(np.max(kappa[i, :]) * 10 + 1) / 10
    min_kappa = math.floor(np.min(kappa[i, :]) * 10 - 1) / 10
    kappa_levels = np.arange(min_kappa, max_kappa, 0.1)

    max_u = math.ceil(max(np.max(u[i, :]), np.max(u_pred[i, :])) * 10 + 2) / 10
    levels = np.arange(0, max_u, 0.3)

    plt.subplot(131)
    ax = plt.gca()
    ax.set_aspect("equal")
    contour = fd.tricontourf(from_numpy(np.reshape(kappa[i, :], (dimension_of_PoI, 1)), fd.Function(V)),
                             axes=ax,
                             levels=kappa_levels)
    fd.triplot(mesh, axes=ax, interior_kw=dict(alpha=0.05))
    plt.colorbar(contour, fraction=0.046, pad=0.04)
    plt.title(str(i) + 'th conductivity field ' + r'$\kappa$')

    plt.subplot(132)
    ax = plt.gca()
    ax.set_aspect("equal")
    contour = fd.tricontourf(from_numpy(np.reshape(u[i, :], (dimension_of_PoI, 1)), fd.Function(V)),
                             axes=ax,
                             levels=levels)
    fd.triplot(mesh, axes=ax, interior_kw=dict(alpha=0.05))
    plt.colorbar(contour, fraction=0.046, pad=0.04)
    plt.title('True ' + str(i) + 'th Test Solution by Firedrake')

    plt.subplot(133)
    ax = plt.gca()
    ax.set_aspect("equal")
    contour = fd.tricontourf(from_numpy(np.reshape(u_pred[i, :], (dimension_of_PoI, 1)), fd.Function(V)),
                             axes=ax,
                             levels=levels)
    fd.triplot(mesh, axes=ax, interior_kw=dict(alpha=0.05))
    plt.colorbar(contour, fraction=0.046, pad=0.04)
    plt.title('Predicted ' + str(i) + 'th Solution by nFEM')

    plt.savefig("results/predicted_solutions/pred_" + str(i) + ".png", dpi=600, bbox_inches='tight')
    plt.close()


# 4.1 Loading best neural network model for inference
model_best = torch.load('results/best_model.pt')
u_pred, kappa = model_best(test_Parameters)

True_u = (test_Observations_synthetic).cpu().detach().numpy().astype(np.float64)
u_pred = (u_pred).cpu().detach().numpy().astype(np.float64)
kappa = (kappa).cpu().detach().numpy().astype(np.float64)

Cases = 1
for sample in range(Cases):
    plot_u(True_u, u_pred, kappa, sample)

# 5. Animation of different test samples
image_list = []
for step in range(Cases):
    image_list.append(imageio.v2.imread("results/predicted_solutions/pred_" + str(step) + ".png"))
imageio.mimwrite('results/animations.gif', image_list, duration=0.5)
