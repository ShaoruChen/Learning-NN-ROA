
import numpy as np
from pympc.geometry.polyhedron import Polyhedron
import learning_ROA.utilities as ut
from learning_ROA.models_lib import Dynamics_to_fit
import learning_ROA.models_lib as ml
import learning_ROA.prob_model as pm
from learning_ROA.prob_model import Problem, polyhedral_set, origin_guard, polyhedral_origin_guard
from learning_ROA.ACCPM import ACCPM_algorithm, gurobi_options, ACCPM_options
import learning_ROA.ACCPM as ac
import matplotlib.pyplot as plt
from learning_ROA.prob_model import Lyapunov_candidate


if __name__ == '__main__':
    nx = 3
    dt = 0.1

    x_min = np.array([-1.0, -1.0, -1.0])
    x_max = np.array([1.0, 1.0, 1.0])
    domain = Polyhedron.from_bounds(x_min, x_max)

    # linearization at the origin
    A = np.array([[-1.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -10.0]])*dt + np.eye(nx)
    dyn_fcn = ml.polynomial_3D
    ref_dyn = Dynamics_to_fit(dyn_fcn, A, dt)

    ##############################################################################################################
    # generate training data
    ##############################################################################################################

    generate_data = False
    if generate_data:
        # generate training data
        resol = 50
        x_train_samples, y_train_samples = ut.generate_training_data(ref_dyn.err_dynamics, domain, resol)
        ut.pickle_file(x_train_samples, 'x_train_samples')
        ut.pickle_file(y_train_samples, 'y_train_samples')

        # generate validation data
        bounds_list = [[x_min[i], x_max[i]] for i in range(nx)]
        x_val_samples = ut.random_unif_sample_from_box(bounds_list, 20000)
        y_val_samples = ut.sample_vector_field(ref_dyn.err_dynamics, x_val_samples)
        ut.pickle_file(x_val_samples, 'x_val_samples')
        ut.pickle_file(y_val_samples, 'y_val_samples')
    else:
        x_train_samples = ut.load_pickle_file('x_train_samples')
        y_train_samples = ut.load_pickle_file('y_train_samples')
        x_val_samples = ut.load_pickle_file('x_val_samples')
        y_val_samples = ut.load_pickle_file('y_val_samples')

    ##############################################################################################################
    # train NN
    ##############################################################################################################

    is_train = False
    keras_nn_model_name = 'dyn_sys_nn'
    if is_train:
        nn_dims = [3, 100, 100, 100, 3]
        val_data = (x_val_samples, y_val_samples)

        nn_model = ut.train_nn_keras(x_train_samples, y_train_samples, nn_dims, num_epochs= 100, batch_size=20, regularizer_weight=1e-4, val_data = val_data)

        ut.save_nn_model(nn_model, keras_nn_model_name)

        nn_model = ut.load_nn_model(keras_nn_model_name)
        nn_model = ut.modify_nn_equilibrium(nn_model)

    else:
        nn_model = ut.load_nn_model(keras_nn_model_name)
        nn_model = ut.modify_nn_equilibrium(nn_model)

    ##############################################################################################################
    # generate nn dynamics
    ##############################################################################################################
    nn_dynamics = pm.nn_net_model(nn_model, domain)
    nn_dynamics_list = [nn_dynamics]
    input_idx_list = [[0, 1, 2]]
    output_idx_list = [[0, 1, 2]]
    nn_dynamics_vector_field = pm.nn_vector_field(nx, nn_dynamics_list, input_idx_list, output_idx_list)

    ##############################################################################################################
    # find ROI
    ##############################################################################################################
    load_ref_ROI = True
    if load_ref_ROI:
        ref_ROI_info = ut.load_pickle_file('ref_ROI_info')
        ref_ROI = ref_ROI_info['ref_ROI']
        vertices = ref_ROI_info['vertices']
    else:
        N_dim = 50
        N_step = 50
        terminal_states, norm_list, init_samples = ut.sample_terminal_states(ref_dyn, domain, N_dim, N_step)
        ref_ROI, vertices = ut.generate_reference_ROI_from_samples(init_samples, norm_list, tol=1.0)
        ref_ROI_info = {'ref_ROI': ref_ROI, 'vertices': vertices}
        ut.pickle_file(ref_ROI_info, 'ref_ROI_info')

    ROI = ut.scale_polyhedron(ref_ROI, 0.9)

    # plot the ROI
    plt.figure()
    ROI.plot()
    ut.plot_samples(vertices)
    plt.title('ROI')

    ##############################################################################################################
    # find err bounds
    ##############################################################################################################
    load_err_bds = True

    if load_err_bds:
        err_bd = ut.load_pickle_file('err_bounds_info')
        gamma_values = err_bd['gamma']
        delta_values = err_bd['delta']
        x_samples = ut.load_pickle_file('ROI_err_bd_sample_x')
        w_samples = ut.load_pickle_file('ROI_err_bd_sample_w')
    else:
        sample_resol = 50
        x_samples, w_samples = ut.diff_from_set(nn_dynamics, ref_dyn.err_dynamics, ROI, sample_resol)
        ut.pickle_file(x_samples, 'ROI_err_bd_sample_x')
        ut.pickle_file(w_samples, 'ROI_err_bd_sample_w')

        x_norm = np.array([np.linalg.norm(x, np.inf) for x in x_samples])
        w_norm = np.array([np.linalg.norm(w, np.inf) for w in w_samples])

        plt.figure()
        plt.scatter(x_norm, w_norm)

        gamma_values = np.logspace(-6, 2, 200)
        delta_values = ut.approx_err_bd_grid_search(x_samples, w_samples, gamma_values, norm_type='inf')
        delta_values[delta_values <= 0] = 0

        err_bd_info = {'gamma': gamma_values, 'delta': delta_values}
        ut.pickle_file(err_bd_info, 'err_bounds_info')

        # view bad samples
        compare_list = np.array([np.linalg.norm(w_samples[i], np.inf) - 1e-6 * np.linalg.norm(x_samples[i], np.inf) for i in
                                 range(x_samples.shape[0])])
        idx = np.where(compare_list > 0.008)
        x_bad = x_samples[idx]
        plt.figure()
        ROI.plot(fill=False, ec='r', linestyle='-.', linewidth=2)
        ut.plot_samples(x_bad)

        # view plot
        plt.figure()
        plt.semilogx(gamma_values, delta_values, 'o-', markersize=3)
        plt.xlabel(r'$\gamma^\prime$')
        plt.ylabel(r'$\delta^\prime$')
        plt.title('regularization 5e-4, horizon = 0, epochs = 30')

    ##########################################################################################################
    # select error bounds
    ##########################################################################################################
    # bd_idx = np.where(min(delta_values + 2 * gamma_values))[0][0]

    gamma_values_select = np.array([0.0])
    delta_values_select = ut.approx_err_bd_grid_search(x_samples, w_samples, gamma_values_select, norm_type='inf')
    delta_values_select[delta_values_select <= 0] = 0

    gamma = gamma_values_select.tolist()
    delta = delta_values_select.tolist()

    # construct disturbance models
    dist = pm.dist_model(gamma, delta, nx, nx)

    dist_models = [dist]
    input_idx_list = [[0, 1, 2]]
    output_idx_list = [[0, 1, 2]]
    dist_vec = pm.dist_vector(nx, dist_models, input_idx_list, output_idx_list)


    ##########################################################################################################
    # simulate trajectories
    ##########################################################################################################
    system = pm.approx_dynamics(A, nn_dynamics_vector_field, dist_vec)

    x_init_list = ROI.vertices

    # Nstep = 50
    # traj_list = []
    #
    # for x_0 in x_init_list:
    #     traj = system.simulate_traj(x_0, Nstep, 'nominal')
    #     traj_list.append(traj)
    #
    # plt.figure()
    # dim = [0, 1]
    # ROI.plot(fill=False, ec='r', linestyle='-.', linewidth=2, residual_dimensions= dim)
    # ut.plot_multiple_traj(traj_list, dim = dim)
    # plt.title('nominal nn dynamics simulation')

    # Nstep = 50
    # ref_system = ref_dyn
    # traj_list = []
    # for x_0 in x_init_list:
    #     traj = ref_system.simulate_traj(x_0, Nstep)
    #     traj_list.append(traj)
    #
    # plt.figure()
    # dim = [0, 2]
    # ROI.plot(fill=False, ec='r', linestyle='-.', linewidth=2, residual_dimensions= dim)
    # ut.plot_multiple_traj(traj_list, dim = dim)
    # plt.title('nonlinear dynamics simulation')

    ##########################################################################################################
    # run the algorithm
    ##########################################################################################################
    # add a guard at the origin

    order = 2
    guard_min = np.array([-0.1, -0.1, -0.05])
    guard_max = np.array([0.1, 0.1, 0.05])
    guard = Polyhedron.from_bounds(guard_min, guard_max)
    X_0 = polyhedral_origin_guard(guard.A, guard.b, ROI)

    # construct stability verification problem
    ROI_set = polyhedral_set(ROI.A, ROI.b)
    problem = Problem(system, ROI_set, order, X_0)

    # find tighter domain estimation
    lbs_0, ubs_0 = ac.find_covering_box(problem, None, order=0)
    lbs_1, ubs_1 = ac.find_covering_box(problem, None, order=1)
    lbs_2, ubs_2 = ac.find_covering_box(problem, None, order=2)

    ref_domain_min = np.minimum(lbs_0, lbs_1, lbs_2)
    ref_domain_max = np.maximum(ubs_0, ubs_1, ubs_2)
    ref_domain = Polyhedron.from_bounds(ref_domain_min, ref_domain_max)

    nn_dynamics_renewed = pm.nn_net_model(nn_model, ref_domain)
    nn_dynamics_list = [nn_dynamics_renewed]
    nn_dynamics_vector_field = pm.nn_vector_field(nx, nn_dynamics_list, input_idx_list, output_idx_list)
    system = pm.approx_dynamics(A, nn_dynamics_vector_field, dist_vec)
    problem = Problem(system, ROI_set, order, X_0)

    # running the algorithm
    alg_options = ACCPM_options(max_iter= 50, tol=1e-7)
    solver_options = gurobi_options(best_obj=-1e-7)
    ACCPM_alg = ACCPM_algorithm(problem, alg_options, solver_options)
    alg_status = ACCPM_alg.ACCPM_main_algorithm()
    ACCPM_alg.save_data('ACCPM_data')
    print(alg_status)

    if alg_status == 'feasible':
        P = ACCPM_alg.candidate_record[-1]
        Lyap_fcn = Lyapunov_candidate(system, order, P)

        # plot the level set on the plan x[0]= 0
        plt.figure()
        ROI.plot(fill=False, ec='r', linestyle='-.', linewidth=2, residual_dimensions = [1, 2])

        xlim = [-1.0, 1.0]
        ylim = [-1.0, 1.0]
        N_dim = 40
        x_range = np.linspace(xlim[0], xlim[1], N_dim)
        y_range = np.linspace(ylim[0], ylim[1], N_dim)
        xx, yy = np.meshgrid(x_range, y_range)

        zz = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                traj = Lyap_fcn.evaluate(np.array([0.0, xx[i, j], yy[i, j]]))
                zz[i, j] = traj.flatten() @ Lyap_fcn.P @ traj.flatten()

        # choose the parameter such that the maximal inscribed level set is plotted
        level = 0.6

        h = plt.contour(xx, yy, zz, [level], colors='b')
        for i, label in enumerate(h.cvalues):
            h.collections[i].set_label('robust ROA')

        # plot the level set on the plan x[2] = 0
        plt.figure()
        ROI.plot(fill=False, ec='r', linestyle='-.', linewidth=2, residual_dimensions=[0, 1])

        xlim = [-1.0, 1.0]
        ylim = [-1.0, 1.0]
        N_dim = 40
        x_range = np.linspace(xlim[0], xlim[1], N_dim)
        y_range = np.linspace(ylim[0], ylim[1], N_dim)
        xx, yy = np.meshgrid(x_range, y_range)

        zz = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                traj = Lyap_fcn.evaluate(np.array([xx[i, j], yy[i, j], 0.0]))
                zz[i, j] = traj.flatten() @ Lyap_fcn.P @ traj.flatten()

        h = plt.contour(xx, yy, zz, [level], colors='b')
        for i, label in enumerate(h.cvalues):
            h.collections[i].set_label('robust ROA')