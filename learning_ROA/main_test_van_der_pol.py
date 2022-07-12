

import numpy as np
from pympc.geometry.polyhedron import Polyhedron
import utilities as ut
# from utilities import unif_sample_from_Polyhedron, generate_training_data, train_nn_keras
from models_lib import Dynamics_to_fit, van_der_pol
from prob_model import dist_model, approx_dynamics, Problem, polyhedral_set, origin_guard
from ACCPM import ACCPM_algorithm, gurobi_options, ACCPM_options
from utilities import load_pickle_file
import matplotlib.pyplot as plt
from prob_model import Lyapunov_candidate

if __name__ == '__main__':
    nx = 2

    x_min = np.array([-3, -3])
    x_max = np.array([3, 3])
    domain = Polyhedron.from_bounds(x_min, x_max)

    # linearization at the origin
    # A = np.array([[0.9988, -0.0488], [0.0488, 0.9500]])
    A = np.array([[0.9988, -0.0488], [0.0488, 0.9500]])*0.5

    van_der_pol_model = Dynamics_to_fit(van_der_pol, A)

    X, Y = ut.generate_training_data(van_der_pol_model.err_dynamics, domain, 100)
    nn_dims = [2,20, 20, 20, 2]
    nn_model = ut.train_nn_keras(X, Y, nn_dims, num_epochs=15, batch_size=20, regularizer_weight= None)
    ut.save_nn_model(nn_model, 'van_der_pol')
    nn_model = ut.load_nn_model('van_der_pol')
    # nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    #
    # score_1 = nn_model.evaluate(X, Y, verbose = 0)

    nn_model = ut.modify_nn_equilibrium(nn_model)
    # nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    # score_2 = nn_model.evaluate(X, Y, verbose = 0)

    ref_dyn = Dynamics_to_fit(van_der_pol, A)

    gamma = 0
    delta = 0
    norm_type = 'inf'
    dist = dist_model(gamma, delta, nx, norm_type)

    system = approx_dynamics(nn_model, dist, A, domain)

    # N_dim = 50
    # N_step = 100
    # terminal_states, norm_list, init_samples = ut.sample_terminal_states(system, domain, N_dim, N_step)
    # ref_ROI, vertices = ut.generate_reference_ROI_from_samples(init_samples, norm_list, tol= 1.0)
    # ref_ROI_info = {'ref_ROI': ref_ROI, 'vertices': vertices}
    # ut.pickle_file(ref_ROI_info, 'vdp_ref_ROI_info')


    # plot trajectories
    traj_list = []
    x_0 = np.array([0.0, -0.01])
    Nstep = 1000
    traj = system.simulate_nominal_traj(x_0, Nstep)
    traj_ref = ref_dyn.simulate_random_traj(x_0, Nstep)
    traj_list.append(traj)
    traj_list.append(traj_ref)
    plt.figure()
    ut.plot_multiple_traj(traj_list)



    # formulate the problem
    ref_ROI_info = ut.load_pickle_file('vdp_ref_ROI_info')
    ref_ROI = ref_ROI_info['ref_ROI']
    vertices = ref_ROI_info['vertices']
    ROI = ut.scale_polyhedron(ref_ROI, 0.5)

    plt.figure()
    ROI.plot()
    ut.plot_samples(vertices)
    plt.title('ROI')
    plt.show()

    ##############################################################################################################
    # functions on step 2
    #############################################################################################################

    x_samples, w_samples = ut.diff_from_set(system, van_der_pol, ROI, 50)
    gamma_values = np.logspace(-4, 2, 100)
    delta_values = ut.approx_err_bd_grid_search(x_samples, w_samples, gamma_values, norm_type='inf')
    delta_values[delta_values <= 0] = 0

    plt.figure()
    plt.semilogx(gamma_values, delta_values, 'o-')


    # reset the disturbance norm
    value_idx = 0
    # gamma = gamma_values[value_idx]
    # delta = delta_values[value_idx]
    gamma, delta = 0, 0
    norm_type = 'inf'
    dist = dist_model(gamma, delta, nx, norm_type)
    system.set_dist(dist)

    # add a guard at the origin
    order = 1
    X_0 = origin_guard(0.05, nx)

    # construct stability verification problem
    ROI_set = polyhedral_set(ROI.A, ROI.b)
    problem = Problem(system, ROI_set, order, X_0)

    # running the algorithm
    alg_options = ACCPM_options(max_iter = 20, tol = 1e-6)
    solver_options = gurobi_options(best_obj = 1e-7)
    ACCPM_alg = ACCPM_algorithm(problem, alg_options, solver_options)
    alg_status = ACCPM_alg.ACCPM_main_algorithm()
    ACCPM_alg.save_data('ACCPM_vdp')
    print(alg_status)

    if alg_status == 'feasible':
        plt.figure()
        ROI.plot( fill = False, ec = 'r', linestyle = '-.', linewidth = 2)
        ut.plot_samples(vertices)

        P = ACCPM_alg.candidate_record[-1]
        Lyap_fcn = Lyapunov_candidate(system, order, P)

        # plot the level set
        xlim = [-1.0, 1.0]
        ylim = [-1.0, 1.0]
        level = 0.2
        Lyap_fcn.plot_level_set(xlim, ylim, level)
        plt.show()


