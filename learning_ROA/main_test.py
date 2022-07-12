import sys
# sys.path.append('.\pympc')

import numpy as np
from pympc.geometry.polyhedron import Polyhedron
import utilities as ut
# from utilities import unif_sample_from_Polyhedron, generate_training_data, train_nn_keras
from models_lib import Dynamics_to_fit, predator_prey
from prob_model import dist_model, approx_dynamics, Problem, polyhedral_set, origin_guard
from ACCPM import ACCPM_algorithm, gurobi_options, ACCPM_options
from utilities import load_pickle_file
import matplotlib.pyplot as plt
from prob_model import Lyapunov_candidate

if __name__ == '__main__':
    nx = 2

    x_min = np.array([-2, -2])
    x_max = np.array([2, 2])
    domain = Polyhedron.from_bounds(x_min, x_max)

    # linearization at the origin
    A = np.array([[0.5, 0], [0, -0.5]])


    pred_prey_model = Dynamics_to_fit(predator_prey, A)
    #
    # X, Y = ut.generate_training_data(pred_prey_model.err_dynamics, domain, 300)
    # nn_dims = [2,50, 50, 50, 2]
    # nn_model = ut.train_nn_keras(X, Y, nn_dims, num_epochs= 5, batch_size= 20, regularizer_weight= 1e-3)
    #
    # ut.save_nn_model(nn_model, 'predator_deep_sparse')
    nn_model = ut.load_nn_model('predator_deep_sparse')

    gamma = 0
    delta = 0
    norm_type = 'inf'
    dist = dist_model(gamma, delta, nx, norm_type)

    system = approx_dynamics(nn_model, dist, A, domain)

    ##############################################################################################################
    # step 1 train a NN
    #############################################################################################################

    # N_dim = 20
    # N_step = 10
    # terminal_states, norm_list, init_samples = ut.sample_terminal_states(system, domain, N_dim, N_step)
    # ref_ROI, vertices = ut.generate_reference_ROI_from_samples(init_samples, norm_list, tol=0.1)
    # ref_ROI_info = {'ref_ROI': ref_ROI, 'vertices': vertices}
    # ut.pickle_file(ref_ROI_info, 'ref_ROI_info')



    # formulate the problem
    ref_ROI_info = ut.load_pickle_file('ref_ROI_info')
    ref_ROI = ref_ROI_info['ref_ROI']
    vertices = ref_ROI_info['vertices']
    ROI = ut.scale_polyhedron(ref_ROI, 0.45)

    plt.figure()
    ROI.plot()
    ut.plot_samples(vertices)
    plt.title('ROI')
    plt.show()

    ##############################################################################################################
    # functions on step 2
    #############################################################################################################

    x_samples, w_samples = ut.diff_from_set(system, predator_prey, ROI, 200)
    gamma_values = np.logspace(-4, 2, 100)
    delta_values = ut.approx_err_bd_grid_search(x_samples, w_samples, gamma_values, norm_type='inf')
    delta_values[delta_values <= 0] = 0

    plt.figure()
    plt.semilogx(gamma_values, delta_values, 'o-', markersize=3)
    plt.xlabel(r'$\gamma^\prime$')
    plt.ylabel(r'$\delta^\prime$')
    plt.title('error bound estimate from sampling')

    traj_list = []
    x_0 = np.array([1.10, 0.13])
    x_1 = np.array([-0.8, -0.2])
    Nstep = 400
    traj = system.simulate_nominal_traj(x_0, Nstep)
    traj_ref = system.simulate_random_traj(x_1, Nstep)
    traj_list.append(traj)
    traj_list.append(traj_ref)
    plt.figure()
    ut.plot_multiple_traj(traj_list)
    # plt.title('predator-prey nominal dyanamics')

    ##############################################################################################################
    # run the algorithm
    #############################################################################################################

    # reset the disturbance norm
    value_idx = 40
    gamma = gamma_values[value_idx]
    delta = delta_values[value_idx]
    norm_type = 'inf'
    dist = dist_model(gamma, delta, nx, norm_type)
    system.set_dist(dist)

    # add a guard at the origin
    order = 2
    radius = 0.01
    X_0 = origin_guard(radius, nx)

    # construct stability verification problem
    ROI_set = polyhedral_set(ROI.A, ROI.b)
    problem = Problem(system, ROI_set, order, X_0)

    # running the algorithm
    alg_options = ACCPM_options(max_iter = 50, tol = 1e-6)
    solver_options = gurobi_options(best_obj = -1e-6)
    ACCPM_alg = ACCPM_algorithm(problem, alg_options, solver_options)
    alg_status = ACCPM_alg.ACCPM_main_algorithm()
    ACCPM_alg.save_data('ACCPM_predator')
    print(alg_status)

    if alg_status == 'feasible':
        plt.figure()
        ROI.plot( fill = False, ec = 'r', linestyle = '-.', linewidth = 2)
        ut.plot_samples(vertices)

        P = ACCPM_alg.candidate_record[-1]
        Lyap_fcn = Lyapunov_candidate(system, order, P)

        # plot the level set
        xlim = [-2.0, 2.0]
        ylim = [-2.0, 2.0]
        level = 0.6
        Lyap_fcn.plot_level_set(xlim, ylim, level)
        plt.show()