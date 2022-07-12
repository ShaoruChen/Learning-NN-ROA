
import sys
sys.path.append('..\pympc')

from pympc.geometry.polyhedron import Polyhedron
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.regularizers import l1
import keras

from pympc.optimization.programs import linear_program
from pympc.plot import plot_state_space_trajectory

import numpy as np
import gurobipy as gp
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader

class DynamicalSystemDataset(Dataset):
    def __init__(self, x_samples, y_samples):
        self.x_samples = x_samples
        self.y_samples = y_samples
        self.num_samples = x_samples.size(0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x_samples[idx, :]
        y = self.y_samples[idx, :]
        return x, y


class torch_nn_model(nn.Module):
    def __init__(self, nn_dims):
        super(torch_nn_model, self).__init__()
        self.dims = nn_dims
        self.L = len(nn_dims) - 2
        self.linears = nn.ModuleList([nn.Linear(nn_dims[i], nn_dims[i+1]) for i in range(len(nn_dims)-1)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i in range(self.L):
            x = F.relu(self.linears[i](x))

        x = self.linears[-1](x)

        return x

# samples along trajectories
def filter_unstable_traj(x_samples, y_samples, threshold = 1e1):
    y_state_norm = torch.linalg.norm(y_samples, np.inf, dim = (1,2))
    idx = (y_state_norm < threshold)
    num_filtered_samples = sum(idx)

    if num_filtered_samples > 0:
        x_samples_filtered = x_samples[idx]
        y_samples_filtered = y_samples[idx]
    else:
        raise ValueError('No samples satisfy the threshold constraint.')

    return x_samples_filtered, y_samples_filtered


def generate_traj_samples(dyn_fcn, init_states, step = 0):
    N = init_states.shape[0]
    nx = init_states.shape[1]

    traj_list = []
    for i in range(N):
        x = init_states[i]
        traj = x
        for j in range(step+1):
            x_next = dyn_fcn(x)
            traj = np.vstack((traj, x_next))
            x = x_next
        traj_list.append(traj)
    x_samples = [traj[0:1, :] for traj in traj_list]
    y_samples = [traj[1:, :] for traj in traj_list]

    x_samples = np.stack(x_samples, axis = 0)
    y_samples = np.stack(y_samples, axis = 0)

    return x_samples, y_samples

def generate_training_data_traj(dyn_fcn, X, N_dim, step = 0):
    init_states = unif_sample_from_Polyhedron(X, N_dim)
    x_samples, y_samples = generate_traj_samples(dyn_fcn, init_states, step)
    return x_samples, y_samples


def criterion(pred_traj, label_traj):
    batch_size = pred_traj.size(0)
    step = pred_traj.size(1)
    label_step = label_traj.size(1)
    if step > label_step:
        warnings.warn('prediction step mismatch')

    slice_step = min(step, label_step)
    nx = pred_traj.size(2)

    label_traj_slice = label_traj[:, :slice_step, :]
    pred_traj_slice = pred_traj[:, :slice_step, :]

    # label_traj_slice_norm = torch.unsqueeze(torch.linalg.norm(label_traj_slice, 2, dim = 2), 2)
    # label_traj_slice = label_traj_slice/label_traj_slice_norm
    # pred_traj_slice = pred_traj_slice/label_traj_slice_norm
    # err = torch.linalg.norm(label_traj_slice.reshape(-1) - pred_traj_slice.reshape(-1), 2)**2/(batch_size*step)

    # err = 0.0
    # for i in range(batch_size):
    #     err += torch.linalg.norm(label_traj_slice[i].reshape(-1) - pred_traj_slice[i].reshape(-1), np.inf)/(torch.linalg.norm(label_traj_slice[i].reshape(-1), np.inf) + 1e-4)
    #
    # err = err/batch_size
    #
    #
    err = torch.linalg.norm(label_traj_slice.reshape(-1) - pred_traj_slice.reshape(-1), 1)/(batch_size*slice_step)

    # err = torch.linalg.norm(label_traj_slice.reshape(-1) - pred_traj_slice.reshape(-1), 2)**2/(pred_traj_slice.reshape(-1).size(0))

    return err

def torch_train_nn(nn_model, dataloader, l1 = None, epochs = 30, step = 5, lr = 1e-4, decay_rate = 1.0, clr = None):

    if clr is None:
        optimizer = optim.Adam(nn_model.parameters(), lr= lr)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay_rate, last_epoch=-1)
        lr_scheduler = lambda t: lr
        cycle = 1
    else:
        lr_base = clr['lr_base']
        lr_max = clr['lr_max']
        step_size = clr['step_size']
        cycle = clr['cycle']
        update_rate = clr['update_rate']
        optimizer = optim.Adam(nn_model.parameters(), lr= lr_max)
        lr_scheduler = lambda t: np.interp([t], [0, step_size, cycle], [lr_base, lr_max, lr_base])[0]

    lr_test = {}
    cycle_loss = 0.0
    cycle_count = 0

    nn_model.train()

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        lr = lr_scheduler((epoch//update_rate)%cycle) # change learning rate every two epochs
        optimizer.param_groups[0].update(lr=lr)

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            batch_size = inputs.size(0)
            x = inputs
            y = nn_model(x)
            traj = y
            for _ in range(step):
                x = y
                y = nn_model(x)
                traj = torch.cat((traj, y), 1)

            loss_1 = criterion(traj, labels)

            # add l1 regularization
            if l1 is not None:
                l1_regularization = 0.0
                for param in nn_model.parameters():
                    '''attention: what's the correct l1 regularization'''
                    l1_regularization += torch.linalg.norm(param.view(-1), 1)
                loss = loss_1 + l1*l1_regularization
            else:
                loss = loss_1

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

            cycle_loss += loss_1.item()

            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        if (epoch + 1) % update_rate == 0:
            lr_test[cycle_count] = cycle_loss / update_rate / len(dataloader)
            print('\n [%d, %.4f] cycle loss: %.6f' % (cycle_count, lr, lr_test[cycle_count]))

            cycle_count += 1
            cycle_loss = 0.0

        # scheduler.step()
    pickle_file(lr_test, 'lr_test')

    print('finished training')
    save_torch_nn_model(nn_model, 'torch_nn_model_dict')
    return nn_model

def train_nn_torch(dataloader, nn_dims, num_epochs= 30, l1 = None, pred_step = 5, lr = 1e-4, decay_rate = 1.0, clr = None, path = 'torch_nn_model_temp'):
    nn_model = torch_nn_model(nn_dims)
    nn_model = torch_train_nn(nn_model, dataloader, l1 = l1, epochs = num_epochs, step = pred_step, lr = lr, decay_rate = decay_rate, clr = clr)
    save_torch_nn_model(nn_model, path)
    return nn_model

def load_torch_nn_model(nn_model, model_param_name):
    nn_model.load_state_dict(torch.load(model_param_name))
    return nn_model

def save_torch_nn_model(nn_model, path):
    torch.save(nn_model.state_dict(), path)

def torch2keras_nn_model(nn_model_torch, file_name = None):
    layers = nn_model_torch.linears
    num_layers = len(layers)
    nn_dims = [layers[0].in_features]
    for i in range(num_layers):
        nn_dims.append(layers[i].out_features)

    # construct keras nn
    keras_nn_model = Sequential()
    nn_size = len(nn_dims)
    keras_nn_model.add(Dense(nn_dims[1], input_dim= nn_dims[0], activation='relu'))
    for ii in range(nn_size - 3):
        keras_nn_model.add(Dense(nn_dims[ii + 2], activation='relu'))
    keras_nn_model.add(Dense(nn_dims[-1]))

    # translate weights
    weights_list = [param.detach().numpy().T for param in nn_model_torch.parameters()]
    keras_nn_model.set_weights(weights_list)

    if file_name is not None:
        save_nn_model(keras_nn_model, file_name)

    return keras_nn_model

def load_pickle_file(file_name):
    with open(file_name, 'rb') as config_dictionary_file:
        data = pickle.load(config_dictionary_file)
    return data

def pickle_file(data, file_name):
    with open(file_name, 'wb') as config_dictionary_file:
          pickle.dump(data, config_dictionary_file)

# random uniform sample from a box
def random_unif_sample_from_box(bounds_list, N):
    # box_list = [[min, max], [min, max], ...]
    box_list = [[item[0], item[1]-item[0]] for item in bounds_list]
    nx = len(box_list)
    rand_matrix = np.random.rand(N, nx)
    samples = np.vstack([rand_matrix[:, i]*box_list[i][1] + box_list[i][0] for i in range(nx)])
    samples = samples.T
    return samples

# uniformly sample from a polyhedron
def unif_sample_from_Polyhedron(X, N_dim, epsilon=None, residual_dim=None):
    # uniformly sample from the Polyhedron X with N_dim grid points on each dimension
    nx = X.A.shape[1]
    if residual_dim is not None:
        X = X.project_to(residual_dim)
    lb, ub = find_bounding_box(X)
    box_grid_samples = grid_sample_from_box(lb, ub, N_dim, epsilon)
    idx_set = [X.contains(box_grid_samples[i, :]) for i in range(box_grid_samples.shape[0])]
    valid_samples = box_grid_samples[idx_set]

    if residual_dim is not None:
        aux_samples = np.zeros((valid_samples.shape[0], 1))
        for i in range(nx):
            if i in residual_dim:
                aux_samples = np.hstack((aux_samples, valid_samples[:, i].reshape(-1, 1)))
            else:
                aux_samples = np.hstack((aux_samples, np.zeros((valid_samples.shape[0], 1))))

        aux_samples = aux_samples[:, 1:]
        return aux_samples

    return valid_samples


def find_bounding_box(X):
    # find the smallest box that contains Polyhedron X
    A = X.A
    b = X.b

    nx = A.shape[1]

    lb_sol = [linear_program(np.eye(nx)[i], A, b) for i in range(nx)]
    lb_val = [lb_sol[i]['min'] for i in range(nx)]

    ub_sol = [linear_program(-np.eye(nx)[i], A, b) for i in range(nx)]
    ub_val = [-ub_sol[i]['min'] for i in range(nx)]

    return lb_val, ub_val


def grid_sample_from_box(lb, ub, Ndim, epsilon=None):
    # generate uniform grid samples from a box {lb <= x <= ub} with Ndim samples on each dimension
    nx = len(lb)
    assert nx == len(ub)

    if epsilon is not None:
        lb = [lb[i] + epsilon for i in range(nx)]
        ub = [ub[i] - epsilon for i in range(nx)]

    grid_samples = grid_sample(lb, ub, Ndim, nx)
    return grid_samples


def grid_sample(lb, ub, Ndim, idx):
    # generate samples using recursion
    nx = len(lb)
    cur_idx = nx - idx
    lb_val = lb[cur_idx]
    ub_val = ub[cur_idx]

    if idx == 1:
        cur_samples = np.linspace(lb_val, ub_val, Ndim)
        return cur_samples.reshape(-1, 1)

    samples = grid_sample(lb, ub, Ndim, idx - 1)
    n_samples = samples.shape[0]
    extended_samples = np.tile(samples, (Ndim, 1))

    cur_samples = np.linspace(lb_val, ub_val, Ndim)
    new_column = np.kron(cur_samples.reshape(-1, 1), np.ones((n_samples, 1)))

    new_samples = np.hstack((new_column, extended_samples))
    return new_samples

def sample_vector_field(dyn_fcn, samples):
    num_samples, nx = samples.shape

    sample = samples[0]
    output = dyn_fcn(sample)
    ny = output.shape[0]

    labels = np.zeros((1,ny))
    for i in tqdm(range(num_samples), desc = 'sample_vector_filed'):
    # for i in range(num_samples):
        x_input = samples[i]
        y = dyn_fcn(x_input)
        labels = np.vstack((labels, y))

    labels = labels[1:,:]
    return labels

def generate_training_data(dyn_fcn, X, N_dim):
    input_samples = unif_sample_from_Polyhedron(X, N_dim)
    labels = sample_vector_field(dyn_fcn, input_samples)
    return input_samples, labels

# neural network training

def train_nn_keras(samples, labels, nn_dims, num_epochs = 100, batch_size = 5, regularizer_weight = None, val_data = None):
    X = samples
    Y = labels

    _, input_size = X.shape
    _, output_size = Y.shape

    model = Sequential()
    nn_size = len(nn_dims)

    if regularizer_weight is not None:
        model.add(Dense(nn_dims[1], kernel_regularizer=l1(regularizer_weight), input_dim=input_size, activation='relu'))
        for ii in range(nn_size - 3):
            model.add(Dense(nn_dims[ii + 2], kernel_regularizer=l1(regularizer_weight), activation='relu'))
        model.add(Dense(output_size))
    else:
        model.add(Dense(nn_dims[1], input_dim=input_size, activation='relu'))
        for ii in range(nn_size - 3):
            model.add(Dense(nn_dims[ii + 2], activation='relu'))
        model.add(Dense(output_size))

    # optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.01)
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer= optimizer, loss= keras.losses.MeanAbsoluteError(), metrics= [keras.metrics.MeanAbsoluteError()])
    # model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

    if val_data is None:
        model.fit(X, Y, epochs=num_epochs, batch_size= batch_size)
    else:
        model.fit(X, Y, epochs=num_epochs, batch_size= batch_size, validation_split = 0.1, validation_data = val_data, verbose = 2)

    scores = model.evaluate(X, Y, verbose=0)
    print('scores: %.10f' % (scores[0]))

    # serialize model to JSON
    save_nn_model(model, 'temp_nn_model')

    return model

def get_nn_params(model):
    """
    Extract the information of the NN model.

    Parameters
    ----------
    model : Keras neural network model

    Returns
    -------
    weights_list : list of numpy.ndarray
        Weights of each layer of the NN.
    dims : list of integer
        Dimensions of the NN layers.
    L : integer
        Number of hidden layers.

    """
    # the weights are saved as the transpose of the true weights
    weights_list = []
    for layer in model.layers:
        weights_list.append(layer.get_weights())

    input_size = model.input_shape[1]
    dims = [input_size]
    for layer in model.layers:
        dims.append(layer.output_shape[-1])

    # extract neural network structure
    L = len(model.layers) - 1
    return weights_list, dims, L

def modify_nn_equilibrium(nn_model):
    """
    Adjust the output layer bias such that nn(0) = 0

    """

    n = nn_model.input_shape[1]
    x_0 = np.zeros((1, n))
    u_0 = nn_model.predict(x_0).flatten()
    weights = nn_model.get_weights()
    weights[-1] -= u_0
    nn_model.set_weights(weights)
    return nn_model

def scale_nn_output(nn_model, scale = 1.0):
    # scale the neural network output by scale
    # usually applied in discretizing continuous time dynamics
    weights  = nn_model.get_weights()
    weights[-2] = scale*weights[-2]
    weights[-1]  = scale*weights[-1]
    nn_model.set_weights(weights)
    return nn_model

def save_nn_model(nn_model, name):

    # serialize model to JSON
    model_json = nn_model.to_json()
    with open(name + '_structure.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    nn_model.save_weights(name + '_weights.h5')


def load_nn_model(name):
    json_file = open(name + '_structure.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name + '_weights.h5')
    print("Loaded model from disk")

    model = loaded_model
    return model






# preactivation bounds

def nn_preactivation_bounds_LP(nn_model, F, h):
    weights_list, dims, L = get_nn_params(nn_model)
    nx = dims[0]
    nu = dims[-1]

    bounds_list = []
    for i in range(L + 1):
        bounds = nn_preactivation_layer_bound_LP(nn_model, F, h, bounds_list, i + 1)
        bounds_list.append(bounds)

    return bounds_list


def nn_preactivation_layer_bound_LP(nn_model, F, h, bounds_list, layer):
    # layer ranges from 1 to L

    weights_list, dims, L = get_nn_params(nn_model)
    nx = dims[0]
    nu = dims[-1]

    gurobi_model = gp.Model('LP')

    z = []
    for i in range(layer):
        z.append(gurobi_model.addVars(dims[i + 1], lb=-gp.GRB.INFINITY, name='z' + str(i)))

    x = []
    for i in range(layer):
        x.append(gurobi_model.addVars(dims[i], lb=-gp.GRB.INFINITY, name='x' + str(i)))

    gurobi_model.update()

    gurobi_model.addConstrs((x[0].prod(dict(zip(range(nx), F[k, :]))) <= h[k] for k in range(F.shape[0])),
                            name='initialization')
    gurobi_model.addConstrs((z[ell][i] == x[ell].prod(
        dict(zip(range(weights_list[ell][0].T.shape[1]), weights_list[ell][0].T[i, :]))) + weights_list[ell][1][i]
                             for ell in range(layer) for i in range(weights_list[ell][0].T.shape[0])),
                            name='linear_map')

    gurobi_model.update()

    N = len(bounds_list)

    assert N == layer - 1
    if N > 0:
        gurobi_model.addConstrs((x[ell + 1][i] >= z[ell][i] for ell in range(N) for i in range(dims[ell + 1])),
                                name='sigma_lb_1')
        gurobi_model.addConstrs((x[ell + 1][i] >= 0 for ell in range(N) for i in range(dims[ell + 1])),
                                name='sigma_lb_2')
        gurobi_model.addConstrs((x[ell + 1][i] <= bounds_list[ell]['ub'][i] / (
                    bounds_list[ell]['ub'][i] - bounds_list[ell]['lb'][i]) * (z[ell][i] - bounds_list[ell]['lb'][i])
                                 for ell in range(N) for i in range(dims[ell + 1]) if
                                 (bounds_list[ell]['ub'][i] > 0 and bounds_list[ell]['lb'][i] < 0)), name='sigma_ub')

        gurobi_model.addConstrs(
            (x[ell + 1][i] == 0 for ell in range(N) for i in range(dims[ell + 1]) if bounds_list[ell]['ub'][i] <= 0),
            name='sigma_ub')
        gurobi_model.addConstrs((x[ell + 1][i] == z[ell][i] for ell in range(N) for i in range(dims[ell + 1]) if
                                 bounds_list[ell]['lb'][i] >= 0), name='sigma_ub')

    gurobi_model.update()
    gurobi_model.setParam('OutputFlag', False)

    lb_vec = np.zeros(dims[layer])
    for i in range(dims[layer]):
        gurobi_model.setObjective(z[layer - 1][i], gp.GRB.MINIMIZE)
        gurobi_model.optimize()
        lb_vec[i] = gurobi_model.objVal

    ub_vec = np.zeros(dims[layer])
    for i in range(dims[layer]):
        gurobi_model.setObjective(z[layer - 1][i], gp.GRB.MAXIMIZE)
        gurobi_model.optimize()
        ub_vec[i] = gurobi_model.objVal

    bounds = {'ub': ub_vec, 'lb': lb_vec}
    return bounds



# polyhedron operation

def scale_polyhedron(X, gamma=1.0):
    A = X.A
    b = X.b

    nx = A.shape[1]
    origin = np.zeros(nx)
    if X.contains(origin):
        assert np.all(b >= np.zeros(b.shape[0]))
        b_tilde = b * gamma
    else:
        warnings.warn('The polyhedron does not contain the origin. \n')
        b_tilde = b

    P = Polyhedron(A, b_tilde)
    return P

def shift_polyhedron(X, vec):
    A = X.A
    b = X.b
    b_new = b + A@vec
    Y = Polyhedron(A, b_new)
    return Y


def LP_over_polyhedron(c, X):
    # min c^T x s.t. x \in X
    A = X.A
    b = X.b
    nx = A.shape[1]
    m = A.shape[0]

    gurobi_model = gp.Model('LP')
    x = gurobi_model.addVars(nx, lb=-gp.GRB.INFINITY)
    gurobi_model.update()

    obj = x.prod(dict(zip(range(nx), c)) )
    gurobi_model.addConstrs( (x.prod(dict(zip(range(nx), A[i])) ) <= b[i] for i in range(m)) )
    gurobi_model.update()

    gurobi_model.setObjective(obj, gp.GRB.MINIMIZE)
    gurobi_model.optimize()

    obj_value = gurobi_model.objVal
    x_opt = list()
    for i in range(nx):
        x_opt.append(x[i].X)
    x_opt = np.array(x_opt)

    return obj_value, x_opt



def dict_from_class(cls):
    return dict( (key, value) for (key, value) in cls.__dict__.items() )

############################################################################################################
# functions to construct ROI
############################################################################################################

def sample_terminal_states(approx_dyn, init_set, N_dim, N_step, samples = None):
    # uniformly sample the initial conditions from a candidate set
    if samples is None:
        samples = unif_sample_from_Polyhedron(init_set, N_dim)
    else:
        samples = samples

    # evolve the NN approximated system for N_step steps and record the state on termination
    nx = approx_dyn.nx
    terminal_states = np.zeros((1, nx))
    print('sampling trajectories over the input set ...')

    for i in tqdm(range(samples.shape[0]), desc = 'sampling_traj'):
        x_0 = samples[i]
        traj = approx_dyn.simulate_traj(x_0, N_step)
        x_N = traj[-1]
        terminal_states = np.vstack((terminal_states, x_N.reshape(1,-1)))

    terminal_states = terminal_states[1:, :]
    norm_list = [np.linalg.norm(item, np.inf) for item in terminal_states]
    return terminal_states, norm_list, samples

def generate_reference_ROI_from_samples(init_samples, norm_list,  tol = 0.1, save_file_name = 'temp_reference_ROI'):
    idx_list = [i for i in range(len(norm_list)) if norm_list[i] <= tol ]
    vertices = [init_samples[idx] for idx in idx_list]
    print('generating reference ROI ...')
    X = Polyhedron.from_convex_hull(vertices)
    print('removing redundant inequalities ...')
    X.remove_redundant_inequalities()

    # save the reference ROI
    pickle_file(X, save_file_name)
    return X, init_samples[idx_list]


# 2D visualization

def plot_polyhedron(D, **kwargs):
    D.plot(**kwargs)

def plot_samples(samples):
    plt.scatter(
        samples[:, 0],
        samples[:, 1],
        color='w',
        edgecolor='k',
        zorder=3
        )


def plot_multiple_traj(x_traj_list, **kwargs):
    num_traj = len(x_traj_list)
    for i in range(num_traj):
        plot_state_space_trajectory(x_traj_list[i], **kwargs)


##############################################################################################################
# functions on step 2
#############################################################################################################
def diff_from_samples(nn_dynamics_model, ref_dyn, x_samples):
    nx = nn_dynamics_model.dim_input
    ny = nn_dynamics_model.dim_output
    traj = np.zeros((1,ny))

    N = x_samples.shape[0]
    for i in tqdm(range(N), desc = 'diff_from_samples'):
    # for i in range(N):
        x = x_samples[i]
        y_ref = ref_dyn(x)
        y_approx = nn_dynamics_model.evaluate(x)
        w = y_ref - y_approx
        traj = np.vstack((traj, w))

    traj = traj[1:, :]
    return traj

def diff_from_set(nn_dynamics_model, ref_dyn, X, N_dim):
    x_samples = unif_sample_from_Polyhedron(X, N_dim)
    w_samples = diff_from_samples(nn_dynamics_model, ref_dyn, x_samples)
    return x_samples, w_samples

#
# def diff_from_samples(approx_dyn, ref_dyn, x_samples):
#     # generate difference samples w
#     nx = approx_dyn.nx
#     traj = np.zeros((1, nx))
#     N = x_samples.shape[0]
#     for i in range(N):
#         x = x_samples[i]
#         y_ref = ref_dyn(x)
#         y_approx = approx_dyn.A@x + approx_dyn.nn_net.predict(x.reshape(1,-1)).flatten()
#         w = y_ref - y_approx
#         traj = np.vstack((traj, w))
#
#     traj = traj[1:, :]
#     return traj

# def diff_from_set(approx_dyn, ref_dyn, X, N_dim):
#     x_samples = unif_sample_from_Polyhedron(X, N_dim)
#     w_samples = diff_from_samples(approx_dyn, ref_dyn, x_samples)
#     return x_samples, w_samples

def approx_err_bd_grid_search(x_samples, w_samples, gamma_values, norm_type = 'inf'):
    if norm_type == 'inf':
        x_norm_list = [np.linalg.norm(item, np.inf) for item in x_samples]
        w_norm_list = [np.linalg.norm(item, np.inf) for item in w_samples]
    elif norm_type == 'one':
        x_norm_list = [np.linalg.norm(item, 1) for item in x_samples]
        w_norm_list = [np.linalg.norm(item, 1) for item in w_samples]
    else:
        raise ValueError('Norms other than inf or one norm are not supported')

    delta_values = []
    for gamma in gamma_values:
        diff = np.array(w_norm_list) - gamma*np.array(x_norm_list)
        delta = diff.max()
        delta_values.append(delta)

    delta_values = np.array(delta_values)
    return delta_values


def view_slice_polytope(val, dim, A, b):
    nx = A.shape[1]
    assert dim <= nx

    b_new = b - A[:, dim]*val
    A_new = np.hstack((A[:, :dim], A[:, dim+1:]))
    X = Polyhedron(A_new, b_new)
    return X




