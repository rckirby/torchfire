import imageio.v2
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fecr import from_numpy
from torch import nn

device = torch.device("cpu")

torch.manual_seed(0)
np.random.seed(0)

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
mesh = fd.UnitSquareMesh(nx, ny)
V = fd.FunctionSpace(mesh, "P", 1)


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

    def forward(self, u):
        """Forward pass before using Firedrake

        Args:
            u (tensor): the train observable vector u_obs

        Returns:
            z (tensor): parameter z vector
        """

        # Mapping vectors u_obs to parameters z
        z_pred = self.Neuralmap2(self.Relu(self.Neuralmap1(u.float())))

        # generate kappa from vectors z
        kappa = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z_pred.float())
        kappa = torch.exp(kappa)

        return z_pred, kappa


model = NeuralNetwork().to(device)


# 4. Plotting function
def plot_u(u, u_pred, i):
    plt.figure(figsize=(13, 6))
    max_u = np.ceil(max(np.max(u[i, :]), np.max(u_pred[i, :])) * 10 + 1) / 10
    min_u = np.floor(min(np.min(u[i, :]), np.min(u_pred[i, :])) * 10 - 1) / 10
    levels = np.arange(min_u, max_u, 0.1)

    plt.subplot(121)
    ax = plt.gca()
    ax.set_aspect("equal")
    contour = fd.tricontourf(from_numpy(np.reshape(u[i, :], (256, 1)), fd.Function(V)), axes=ax, levels=levels)
    fd.triplot(mesh, axes=ax, interior_kw=dict(alpha=0.05))
    plt.colorbar(contour, fraction=0.046, pad=0.04)
    plt.title('True ' + str(i) + 'th conductivity field ' + r'$\kappa$')

    plt.subplot(122)
    ax = plt.gca()
    ax.set_aspect("equal")
    contour = fd.tricontourf(from_numpy(np.reshape(u_pred[i, :], (256, 1)), fd.Function(V)), axes=ax, levels=levels)
    fd.triplot(mesh, axes=ax, interior_kw=dict(alpha=0.05))
    plt.colorbar(contour, fraction=0.046, pad=0.04)
    plt.title('Predicted ' + str(i) + 'th conductivity field ' + r'$\kappa$ by TNet-TorchFire')

    plt.savefig("results/predicted_solutions/pred_" + str(i) + ".png", dpi=600, bbox_inches='tight')
    plt.close()


# 4.1 Loading best neural network model for inference
model_best = torch.load('results/best_model.pt')
z_pred, kappa_pred = model_best(test_Observations)

kappa_pred = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z_pred.float())
kappa_true = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), test_Parameters.float())

kappa_pred = kappa_pred.cpu().detach().numpy().astype(np.float64)
kappa_true = kappa_true.cpu().detach().numpy().astype(np.float64)

Cases = 1
for sample in range(Cases):
    plot_u(kappa_true, kappa_pred, sample)

# 5. Animation of different test samples
image_list = []
for step in range(Cases):
    image_list.append(imageio.v2.imread("results/predicted_solutions/pred_" + str(step) + ".png"))
imageio.mimwrite('results/animations.gif', image_list, duration=0.5)
