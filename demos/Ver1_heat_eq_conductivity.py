import firedrake
# We need to import firedrake_adjoint to fire up the taping!
import firedrake_adjoint  # noqa
import torch
from firedrake import (DirichletBC, FunctionSpace, SpatialCoordinate, Constant,
                       TestFunction, UnitSquareMesh, assemble, dx, grad, inner)
from torchfire import fd_to_torch
from path import Path
import numpy as np
import pandas as pd

from torch import nn
from scipy import sparse

# device = torch.device('cuda')
# device = torch.device("cuda:0")
device = torch.device('cpu')

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
epochs = 2000
neurons = 1000

# ! 0.1 Using Wandb to upload the approach
# filename = 'Heat_TorchFire_#train_' + str(num_train) +'_to_' + str(num_train * repeat_fac) + '_LR_' + str(int(learning_rate))  + '_batch_' + str(batch_size) + '_neurons_' + str(neurons)
# import wandb
# wandb.init(project="Torch_Fire", entity="hainguyenpho", name = filename)
# wandb.config.problem = 'Heat_TorchFire'
# wandb.config.batchsize = batch_size
# wandb.config.learning_rate = learning_rate
# wandb.config.database = num_train

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
dimension_of_PoI = (n + 1) ** 2  # external force field
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

Prematrix = torch.tensor(pre_mat_stiff_sparse.toarray()).to(device)
load_f = torch.tensor(load_vector).to(device)


#! 1.3 Firedrake and Fenics switch matrix
Fenics_to_Fridrake_mat = torch.tensor(np.reshape(pd.read_csv('data/Fenics_to_Firedrake' + '.csv').to_numpy(), ((n+1)**2, (n+1)**2))).to(device)

def Fenics_to_Fridrake(u):
    # Fenics_to_Fridrake_mat @ u
    return torch.einsum('ij, bj -> bi', Fenics_to_Fridrake_mat.float(), u.float())

def Fridrake_to_Fenics(u):
    # Fenics_to_Fridrake_mat.T @ u
    return torch.einsum('ij, bi -> bj', Fenics_to_Fridrake_mat.float(), u.float())
    

#! 1.4 Impose boundary trick
Operator = np.zeros((210, 256))
i = 0
for j in free_index:
    Operator[i, j] = 1
    i += 1
Operator = torch.Tensor(Operator).to(device)

#! 1.5 TorchFire Mesh definition
mesh = UnitSquareMesh(15, 15)
V = FunctionSpace(mesh, "P", 1)
bc = DirichletBC(V, 0, (1, 2, 3))
templates = (firedrake.Function(V), firedrake.Function(V))

def numpy_formatter(np_array):
    return np.array2string(np_array, formatter={'float': lambda x: f'{x:.6f}'})

# This assumes that e^kappa will be computed outside of firedrake
# and stuffed into a piecewise linear FD function.
def assemble_firedrake(u, expkappa):
    x = SpatialCoordinate(mesh)
    v = TestFunction(u.function_space())
    f = Constant(20.0)

    return assemble(inner(expkappa * grad(u), grad(v)) * dx - inner(f, v) * dx, bcs=bc)

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
        kappa = torch.exp(kappa)
        kappa = Fenics_to_Fridrake(kappa)
        
        return u, kappa
    
    def SolverTorch(self, u, kappa):
        """This generates the solution for all the samples within a batch given a batch of
        predicted solutions `u` and corresponding `exp(kappa)`

        Args:
            u (tensor): predicted solutions of neural network
            kappa (tensor): the corresponding kappa to predicted solutions

        Returns:
             torch.Tensor: the solutions for each pde in the batch
        """
        diff_solver = fd_to_torch(assemble_firedrake, templates, "solverTorch").apply
        outputs = torch.zeros_like( u )
        batch_size = u.shape[0] 
        for i, (u_, kappa_) in enumerate(zip(u, kappa)):
            output = diff_solver(u_, kappa_)
            e_i = torch.zeros_like(u, dtype=torch.float64)
            e_i[i, :] += 1.0
            outputs += output * e_i
        return outputs    

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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
functional = nn.MSELoss()


def train_loop(model, optimizer, z, u_train_true, load_f, functional):
    """This is training loop, optimizing for neural network parameters

    Args:
        model (Pytorch functional): the neural network
        optimizer (Pytorch functional): ADAM optimizer
        z (tensort): vectors z
        u_train_true (tensors): the true solutions w.r.t. vectors
        load_f (tensort): the transformed load vector, that captures BCs as well
        functional (Pytorch functional): Mean square error

    Returns:
        loss_train (scalar): the sum of Residuals
        train_u_acc (scalar): the mean square error of predicted solutions
    """
    loss_train = 0
    for batch in range(int(num_train / batch_size)):
        u_train_pred, kappa = model(z[(batch) * batch_size:(batch + 1) * batch_size, :])
        #Residuals = model.FireDrake(u_train_pred, kappa, load_f)

        # This computes the residual loss given the tensors exp(kapp) and the neural network that generates the solution u_nn
        residuals = model.ResidualTorch(u=u_train_pred, kappa=kappa)
        loss = residuals / batch_size
        loss_train += loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    u_train_pred, _ = model(z)
    train_u_acc = functional(u_train_pred, Fenics_to_Fridrake(u_train_true.squeeze()))

    return loss_train, train_u_acc


def test_loop(model, z_test, u_test_true, functional):
    """This is test functions

    Args:
        model (Pytorch model): model is forward function in network class (default)
        z_test (torch tensor): vectors z (n_test x number of eigen modes)
        u_test_true (torch tensor): true solutions w.r.t. vectors z
        functional (Pytorch function): Mean Square Error

    Returns:
        scalar: the mean square error of predicted solutions
    """
    with torch.no_grad():
        u_test_pred, _ = model(z_test)
        test_u_acc = 0
        # test_u_acc += functional(u_test_pred, Fenics_to_Fridrake(u_test_true.squeeze())) / functional(u_test_pred*0, Fenics_to_Fridrake(u_test_true.squeeze()))
        for i in range(500):
            test_u_acc += functional(u_test_pred[i,:], Fenics_to_Fridrake(u_test_true.squeeze())[i,:]) / functional(u_test_pred[i,:]*0, Fenics_to_Fridrake(u_test_true.squeeze())[i,:])

        # import pdb
        # pdb.set_trace()
        
    return test_u_acc/500


# ! 3. Training process
TRAIN_LOSS, TEST_ACC = [], []
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loss, train_u_acc = train_loop(model, optimizer, train_Parameters, train_Observations_synthetic, load_f,
                                         functional)
    test_u_acc = test_loop(model, test_Parameters, test_Observations_synthetic, functional)

    str_test_u_acc = numpy_formatter(test_u_acc.cpu().detach().numpy())
    str_train_u_acc = numpy_formatter(train_u_acc.cpu().detach().numpy())
    str_train_loss = numpy_formatter(train_loss.cpu().detach().numpy()[0])
    
    print(f"Test Acc:  {str_test_u_acc} Train Acc: {str_train_u_acc}  Train loss {str_train_loss} \n")
    
    # Save
    test_u_acc_old = 100
    if test_u_acc < test_u_acc_old:
        torch.save(model, 'best_model.pt')
        test_u_acc_old = test_u_acc
        
    TRAIN_LOSS.append(train_loss.cpu().detach().numpy())
    TEST_ACC.append(test_u_acc.cpu().detach().numpy())
    
    # writer.add_scalar("Loss/train", train_loss, t)
    # wandb.log({"Test ACC": float(test_u_acc), "Train ACC": float(train_u_acc), "Train loss": float(train_loss)})

# writer.flush()
# writer.close()
pd.DataFrame(np.asarray(TRAIN_LOSS)).to_csv(Path('data/TRAIN_LOSS.csv'), index=False)
pd.DataFrame(np.asarray(TEST_ACC)).to_csv(Path('data/TEST_ACC.csv'), index=False)
    
print("Done!")


