from mpi4py import MPI

from torchfire import torchfireParallelMap, torchfireExit, torchfireBuildFactory

import numpy as np
import pandas as pd
import torch
from torch import nn

device = torch.device("cpu")

# STEP 3. Building neural network
class NeuralNetwork(nn.Module):
    def __init__(self, neurons, num_observation, num_train, num_truncated_series, Eigen, Sigma):
        super(NeuralNetwork, self).__init__()
        self.num_observation = num_observation
        self.num_train = num_train
        self.Eigen = Eigen
        self.Sigma = Sigma
        
        self.Neuralmap1 = nn.Linear(self.num_observation, neurons)
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
        kappa = torch.einsum('ij,bj -> bi', torch.matmul(self.Eigen, self.Sigma).float(), z_pred.float())
        kappa = torch.exp(kappa)

        return z_pred, kappa

def load_data(name, target_shape=(-1,)):
    return torch.tensor(np.reshape(pd.read_csv(name).to_numpy(), target_shape)).to(device)

def solverTorch(kappa, train_observations, comm):
    """This generates the solution for all the samples within a batch given a batch of
        predicted solutions `exp(u)`
    
        Args:
            kappa (tensor): predicted parameters obtained by neural network
            train_observations (tensor): Sparse Observations
    
        Returns:
             scalar : mean square error between reproduced observations and sparse true observations
    """
    
    loss_mc = torch.zeros(1, device=device)
    losses = torchfireParallelMap("mcLoss", list(zip(train_observations, kappa)), comm)
    
    for l in losses:
        loss_mc += l
    return loss_mc


def train_loop(model, optimizer, z, u_train_true, alpha,
               num_truncated_series, num_train, num_observation, comm):
    u_obs_batch, z_batch = u_train_true.float(), z.float()

    z_pred, kappa = model(u_obs_batch)
    print("computing ML loss")
    loss_ml = torch.mean(torch.square(z_pred - z_batch * 0)) * num_truncated_series

    print("computing MC loss")
    loss_mc = solverTorch(kappa, u_obs_batch, comm) * num_observation / num_train
    loss = loss_ml + alpha * loss_mc

    optimizer.zero_grad()
    #loss.backward()
    loss_mc.backward()
    print(model.Neuralmap1.weight.grad)
    optimizer.step()

    return loss

def test_loop(model, z_test, u_test_true, Eigen, Sigma):
    with torch.no_grad():
        z_test_pred, _ = model(u_test_true)
        kappa_pred = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z_test_pred)
        kappa_true = torch.einsum('ij,bj -> bi', torch.matmul(Eigen, Sigma).float(), z_test.float())

    return torch.mean(
        torch.linalg.vector_norm(kappa_pred - kappa_true, dim=-1) ** 2 / torch.linalg.vector_norm(kappa_true,
                                                                                                  dim=-1) ** 2)

def numpy_formatter(np_array):
    return np.array2string(np_array, formatter={'float': lambda x: f'{x:.6f}'})


def run(factory_functions: dict, ensemble_comm) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Set physics parameters and build workers
    params = {}
    params['nx'] = 15
    params['ny'] = 15
    params['num_observation'] = 10
    params['dimension_of_PoI'] = (params['nx'] + 1) * (params['ny'] + 1)
    params['num_truncated_series'] = 15  # dimension of z vector
    params['device'] = device

    # build the MC loss function on each worker with given params
    torchfireBuildFactory("mcLoss", params, ensemble_comm)
    
    # Set the training parameters
    num_train_ultimate = 10000
    num_train = num_train_ultimate 
    num_train = 100
    num_test = 500
    repeat_fac = 1  # Keep it 1 for now!

    # Set the model and optimizer parameters
    learning_rate = 1e-3
    batch_size = num_train
    epochs = 2
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
    # 1.1 Loading train and test data
    train_observations_synthetic = load_data('data/Training_Sparse_Solutions_u.csv',
                                             (num_train_ultimate, -1))
    train_observations_synthetic = train_observations_synthetic[:num_train, :].repeat(repeat_fac, 1)
    
    train_parameters = load_data('data/Training_KL_Expansion_coefficients.csv',
                                 (num_train_ultimate, -1))
    train_parameters = train_parameters[:num_train, :].repeat(repeat_fac, 1)
    
    test_observations_synthetic = load_data('data/Test_Sparse_Solutions_u.csv', (num_test, -1))
    test_parameters = load_data('data/Test_KL_Expansion_coefficients.csv', (num_test, -1))
    
    # Add noise to data
    train_observations = (train_observations_synthetic
                          + torch.normal(0, 1, train_observations_synthetic.shape).to(device)
                          * torch.unsqueeze(
                              torch.max(train_observations_synthetic,
                                        axis=1)[0].to(device), dim=1)
                          * noise_level)
    test_observations = (test_observations_synthetic
                         + torch.normal(0, 1, test_observations_synthetic.shape).to(device)
                         * torch.unsqueeze(
                             torch.max(test_observations_synthetic,
                                       axis=1)[0].to(device), dim=1)
                         * noise_level)


    # 1.2 Loading eigenvalues, eigenvectors, observed indices, Degree of Freedom indices, and sparse observables    
    Eigen = load_data('data/Eigen_vector.csv',
                      (params['dimension_of_PoI'],
                       params['num_truncated_series']))
    Sigma = load_data('data/Eigen_value_data.csv',
                      (params['num_truncated_series'],
                       params['num_truncated_series']))

    model = NeuralNetwork(neurons, params['num_observation'],
                          num_train, params['num_truncated_series'], Eigen, Sigma).to(device)

    # STEP 5. Training process
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    TRAIN_LOSS, TEST_ACC = [], []
    for t in range(epochs):
        
        train_loss = train_loop(model, optimizer, train_parameters,
                                train_observations, alpha,
                                params['num_truncated_series'],
                                num_train, params['num_observation'], ensemble_comm)
        test_u_acc = test_loop(model, test_parameters, test_observations, Eigen, Sigma)
        
        str_test_u_acc = numpy_formatter(test_u_acc.cpu().detach().numpy())
        str_train_loss = numpy_formatter(train_loss.cpu().detach().numpy()[0])
        
        if t % 1 == 0:
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

    torchfireExit(ensemble_comm)
    
    print("Done!")

