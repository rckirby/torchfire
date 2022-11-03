import numpy as np
import pandas as pd

import torch
from torch import nn
from scipy import sparse

# device = torch.device('cuda')
device = torch.device("cpu")

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

# %%
#! 0. Initial parameters and training parameters
num_train2 = 10000
num_train = 600
num_test = 500
repeat_fac = 1 # Keep it 1 for now!
# %%
learning_rate = 1e-2
batch_size = 200
epochs = 1000
neurons = 1000

#! 0.1 Using Wandb to upload the approach
# filename = 'Heat_TorchFire_#train_' + str(num_train) +'_to_' + str(num_train * repeat_fac) + '_LR_' + str(int(learning_rate))  + '_batch_' + str(batch_size) + '_neurons_' + str(neurons)
# import wandb
# wandb.init(project="Torch_Fire", entity="hainguyenpho", name = filename)
# wandb.config.problem = 'Heat_TorchFire'
# wandb.config.batchsize = batch_size
# wandb.config.learning_rate = learning_rate
# wandb.config.database = num_train

#! 1. Loading data by pandas
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
#! 1.2 Loading eigenvalues, eigenvectors
#? 1.2 Load Eigenvalue, Eigenvectors, observed indices, prematrices
#? Physical model information
n = 15
num_observation = 10  # number of observed points
dimension_of_PoI = (n + 1)**2  # external force field
num_truncated_series = 15

df_Eigen = pd.read_csv('data/Eigenvector_data' + '.csv')
df_Sigma = pd.read_csv('data/Eigen_value_data' + '.csv')

Eigen = torch.tensor(np.reshape(df_Eigen.to_numpy(),(dimension_of_PoI, num_truncated_series))).to(device)
Sigma = torch.tensor(np.reshape(df_Sigma.to_numpy(),(num_truncated_series, num_truncated_series))).to(device)

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


Operator = np.zeros((210,256))
i = 0
for j in free_index:
        Operator[i,j] = 1
        i += 1 

Operator = torch.Tensor(Operator).to(device)


#! 2. Building neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.Neuralmap1 = nn.Linear(15,neurons)
        self.Relu = nn.ReLU()
        self.Neuralmap2 = nn.Linear(neurons,210)
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
        
        #? Mapping vectors z to nodal solutions at free nodes (excludes the boundary nodes)
        u = self.Neuralmap2(self.Relu(self.Neuralmap1(z.float())))
        
        #? THIS IS IMPOSED BOUNDARY CONDITIONS
        u = torch.einsum('ij, bi -> bj', Operator, u)
        
        #? generate kappa from vectors z
        kappa = torch.einsum('ij,bj -> bi',torch.matmul(Eigen, Sigma).float(), z.float())
        kappa = torch.exp(kappa)
        
        return u, kappa
    
    def FireDrake(self, u, kappa, load_f):
        """This is API FireDrake/TorchFire should be replaced

        Args:
            u (tensor): predicted solutions of neural network
            kappa (tensor): the corresponding kappe to predicted solutions

        Returns:
            scalar : the Residuals of all samples
        """ 
        
        #? This is trick that is developed to avoid using FireDrake/TorchFire
        A_kappa = torch.einsum('ij, bj -> bi', Prematrix.float(), kappa)
        A_kappa = torch.reshape(A_kappa, (u.shape[0], 256, 256))
        f_out = torch.einsum('bjk, bk ->bj', A_kappa, u)[:, free_index]
        
        return torch.sum((f_out - load_f)**2)


from fecr import from_numpy, to_numpy
import matplotlib.pyplot as plt
import firedrake as fd
from firedrake import *

n= 15
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "P", 1)

#! 1.3 Firedrake and Fenics switch matrix
Fenics_to_Fridrake_mat = torch.tensor(np.reshape(pd.read_csv('data/Fenics_to_Firedrake' + '.csv').to_numpy(), ((n+1)**2, (n+1)**2))).to(device)

def Fenics_to_Fridrake(u):
    # Fenics_to_Fridrake_mat @ u
    return torch.einsum('ij, bj -> bi', Fenics_to_Fridrake_mat.float(), u.float())

def Fridrake_to_Fenics(u):
    # Fenics_to_Fridrake_mat.T @ u
    return torch.einsum('ij, bi -> bj', Fenics_to_Fridrake_mat.float(), u.float())

def plot_u(u, i, filename):
    # plot saving figure    
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_aspect("equal")
    l = tricontourf(from_numpy(np.reshape(u[i, :], (256,1)), fd.Function(V)), axes=ax)
    triplot(mesh, axes=ax, interior_kw=dict(alpha=0.05))
    plt.colorbar(l,fraction=0.046, pad=0.04)
    
    plt.title(filename)
    plt.savefig("Predicted_solutions/" + filename + str(i) + ".png", dpi=600, bbox_inches='tight')
    plt.savefig("Predicted_solutions/" + filename + ".png", dpi=600, bbox_inches='tight')
    plt.close()

True_u = Fenics_to_Fridrake(test_Observations_synthetic).cpu().detach().numpy().astype(np.float64)

# import pdb
# pdb.set_trace()

# Load
model_best = torch.load('best_model.pt')
u_pred, _ = model_best(test_Parameters)
u_pred = Fenics_to_Fridrake(u_pred).cpu().detach().numpy().astype(np.float64)

sample = 4
plot_u(True_u, sample, 'True')
plot_u(u_pred, sample, 'Pred')

# import pdb
# pdb.set_trace()
