import numpy as np
import pandas as pd

import torch
from torch import nn
from scipy import sparse

from pathlib import Path

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
learning_rate = 1e-3
batch_size = 200
epochs = 10
neurons = 500

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
        
#! 3. Training functions
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
        
        u_train_pred, kappa = model(z[(batch)*batch_size:(batch+1)*batch_size, :])
        Residuals = model.FireDrake(u_train_pred, kappa, load_f)
        
        loss = Residuals/batch_size
        loss_train += loss
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

    u_train_pred, _ = model(z)
    train_u_acc =  functional(u_train_pred, u_train_true.squeeze())
    
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
        test_u_acc =  functional(u_test_pred, u_test_true.squeeze())
        
    return test_u_acc

#! 3. Training process
TRAIN_LOSS, TEST_ACC = [], []
for t in range(epochs):
    
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_u_acc = train_loop(model, optimizer, train_Parameters, train_Observations_synthetic, load_f, functional)
    test_u_acc = test_loop(model, test_Parameters, test_Observations_synthetic, functional)
    
    # Save
    test_u_acc_old = 100
    if test_u_acc < test_u_acc_old:
        torch.save(model, 'best_model.pt')
        test_u_acc_old = test_u_acc
    
    print(f"Test Acc: {test_u_acc:>1e} Train Acc: {train_u_acc:>1e}  Train loss {train_loss:>1e} \n")
    
    TRAIN_LOSS.append(train_loss.cpu().detach().numpy())
    TEST_ACC.append(test_u_acc.cpu().detach().numpy())
    
    # writer.add_scalar("Loss/train", train_loss, t)
    
    # wandb.log({"Test ACC": float(test_u_acc), "Train ACC": float(train_u_acc), "Train loss": float(train_loss)})

# writer.flush()
# writer.close()
# import pdb
# pdb.set_trace()

pd.DataFrame(np.asarray(TRAIN_LOSS)).to_csv(Path('data/TRAIN_LOSS.csv'), index=False)
pd.DataFrame(np.asarray(TEST_ACC)).to_csv(Path('data/TEST_ACC.csv'), index=False)

print("Done!")



