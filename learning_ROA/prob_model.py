
import sys
sys.path.append("..\pympc")

import gurobipy as gp
import numpy as np

from pympc.geometry.polyhedron import Polyhedron
from .utilities import get_nn_params, nn_preactivation_bounds_LP, LP_over_polyhedron
import matplotlib.pyplot as plt

class nn_net_model:
	'''multiple inputs single output NN model'''
	def __init__(self, nn_model, domain):
		self.nn_model = nn_model
		weights_list, dims, L = get_nn_params(nn_model)
		self.dims = dims
		self.dim_input = dims[0]
		self.dim_output = dims[-1]
		self.num_hidden_layers = L
		self.weights_list = weights_list

		# add the domain of the NN, and domain is pympc.Polyhedron
		self.domain = domain
		bounds_list = nn_preactivation_bounds_LP(nn_model, domain.A, domain.b)
		ubs = [bounds['ub'] for bounds in bounds_list]
		lbs = [bounds['lb'] for bounds in bounds_list]
		self.ubs = ubs
		self.lbs = lbs

	def set_bounds(self, lbs, ubs):
		# set the neuron-wise lower and upper bounds for MIP description of the NN
		self.lbs = lbs
		self.ubs = ubs

	def count_stable_relus(self):
		ubs = self.ubs
		lbs = self.lbs
		L = len(ubs) - 1

		num_stable_relus = []
		for i in range(L):
			ub = ubs[i]
			lb = lbs[i]
			num_stable_ub = sum(ub < 0)
			num_stable_lb = sum(lb[ub >= 0] > 0)
			num_stable_relus.append(num_stable_lb + num_stable_ub)

		return sum(num_stable_relus), num_stable_relus


	def evaluate(self, x):
		# evaluate the NN
		y = self.nn_model.predict(x.reshape(1, -1), verbose = 0).flatten()
		return y

	def add_gurobi_constrs(self, gurobi_model, input_ids, output_ids, mark):
		# input_ids: list of strings, names of input variables; output_ids: list of strings, names of output variables

		x = list()
		for id in input_ids:
			x.append(gurobi_model.getVarByName(id))

		y = list()
		for id in output_ids:
			y.append(gurobi_model.getVarByName(id))

		# input and output dimensions
		nx = self.dim_input
		ny = self.dim_output

		assert nx == len(input_ids)
		assert ny == len(output_ids)

		# get NN dimensions
		dims = self.dims
		L = self.num_hidden_layers
		weights_list = self.weights_list
		ub_neurons = self.ubs
		lb_neurons = self.lbs

		# add auxiliary continuous variables
		z = []
		for idx in range(L + 1):
			z.append(gurobi_model.addVars(dims[idx], lb=-gp.GRB.INFINITY, name='z' + '_step_' + str(mark) + '_' + str(idx)))

		gurobi_model.update()

		# add binary variables
		t = []
		for idx in range(L):
			t.append(gurobi_model.addVars(dims[idx + 1], name='t' + '_step_' + str(mark) + '_' + str(idx), vtype=gp.GRB.BINARY))

		gurobi_model.update()

		# model neural network
		gurobi_model.addConstrs((z[0][i] == x[i] for i in range(nx)), name='initialization')

		gurobi_model.addConstrs((y[i] == z[L].prod(
			dict(zip(range(weights_list[L][0].T.shape[1]), weights_list[L][0].T[i, :]))) + weights_list[L][1][i]
								 for i in range(ny)), name='outputConstr' + '_step_' + str(mark))

		gurobi_model.addConstrs((z[ell + 1][i] >= z[ell].prod(
			dict(zip(range(weights_list[ell][0].T.shape[1]), weights_list[ell][0].T[i, :]))) + weights_list[ell][1][i]
								 for ell in range(L) for i in range(len(z[ell + 1]))),
								name='binary_1' + '_step_' + str(mark))

		gurobi_model.addConstrs((z[ell + 1][i] >= 0 for ell in range(L) for i in range(len(z[ell + 1]))),
								name='binary_3' + '_step_' + str(mark))

		gurobi_model.addConstrs((z[ell + 1][i] <= z[ell].prod(
			dict(zip(range(weights_list[ell][0].T.shape[1]), weights_list[ell][0].T[i, :]))) + weights_list[ell][1][i]
								 - lb_neurons[ell][i] * (1 - t[ell][i]) for ell in range(L) for i in
								 range(len(z[ell + 1]))   ), name='binary_2' + '_step_' + str(mark))

		gurobi_model.addConstrs(
			(z[ell + 1][i] <= ub_neurons[ell][i] * t[ell][i] for ell in range(L) for i in range(len(z[ell + 1]))),
			name='binary_4' + '_step_' + str(mark) )

		gurobi_model.update()


		###########################################
		# new codes: check if ReLU is unstable first
		###########################################

		# gurobi_model.addConstrs((z[0][i] == x[i] for i in range(nx)), name='initialization')
		#
		# gurobi_model.addConstrs((y[i] == z[L].prod(
		# 	dict(zip(range(weights_list[L][0].T.shape[1]), weights_list[L][0].T[i, :]))) + weights_list[L][1][i]
		# 						 for i in range(ny)), name='outputConstr' + '_step_' + str(mark))
		#
		# gurobi_model.addConstrs(( z[ell+1][i] == z[ell].prod(dict(zip(range(weights_list[ell][0].T.shape[1]), weights_list[ell][0].T[i, :]))) + weights_list[ell][1][i]
		# 						  for ell in range(L) for i in range(len(z[ell + 1])) if lb_neurons[ell][i] >= 0 ) )
		#
		# gurobi_model.addConstrs(( z[ell+1][i] == 0 for ell in range(L) for i in range(len(z[ell + 1])) if ub_neurons[ell][i] <= 0 ) )
		#
		# gurobi_model.addConstrs((z[ell + 1][i] >= z[ell].prod(
		# 	dict(zip(range(weights_list[ell][0].T.shape[1]), weights_list[ell][0].T[i, :]))) + weights_list[ell][1][i]
		# 						 for ell in range(L) for i in range(len(z[ell + 1])) if (lb_neurons[ell][i] <0 and ub_neurons[ell][i] >0) ),
		# 						name='binary_1' + '_step_' + str(mark))
		#
		# gurobi_model.addConstrs((z[ell + 1][i] >= 0 for ell in range(L) for i in range(len(z[ell + 1])) if (lb_neurons[ell][i] <0 and ub_neurons[ell][i] >0) ),
		# 						name='binary_3' + '_step_' + str(mark))
		#
		# gurobi_model.update()
		#
		# for ell in range(L):
		# 	for i in range(len(z[ell+1])):
		# 		if (lb_neurons[ell][i] <0 and ub_neurons[ell][i] >0):
		# 			t = gurobi_model.addVar(vtype=gp.GRB.BINARY)
		# 			gurobi_model.addConstr(z[ell + 1][i] <= z[ell].prod(dict(zip(range(weights_list[ell][0].T.shape[1]), weights_list[ell][0].T[i, :]))) + weights_list[ell][1][i]
		# 						  - lb_neurons[ell][i] * (1 - t))
		# 			gurobi_model.addConstr(z[ell + 1][i] <= ub_neurons[ell][i]*t)
		#
		# gurobi_model.update()

		return gurobi_model

class dist_model:
	def __init__(self, gamma, delta, nx, ny, norm_type = 'inf'):
		# describe the disturbance model ||w|| <= gamma ||x|| + delta
		if not isinstance(gamma, list):
			self.gamma = [gamma]
		else:
			self.gamma = gamma

		if not isinstance(delta, list):
			self.delta = [delta]
		else:
			self.delta = delta

		self.nx = nx
		self.ny = ny
		self.norm_type = norm_type

	def random_sample(self, x):
		# randomly sample a disturbance from the admissible set

		# ell_inf ball
		if self.norm_type == 'inf':
			radius = self.gamma[0] * np.linalg.norm(x, np.inf) + self.delta[0]
			w = (2 * np.random.rand(self.ny) - 1.0) * radius
			return w
		elif self.norm_type == 'one':
			raise ValueError('Not implemented')
		else:
			raise ValueError('Norm type not supported')

	def add_gurobi_constrs(self, gurobi_model, x_ids, w_ids, mark):
		nx = self.nx
		ny = self.ny

		x = list()
		for id in x_ids:
			x.append(gurobi_model.getVarByName(id))

		w = list()
		for id in w_ids:
			w.append(gurobi_model.getVarByName(id))

		abs_x = gurobi_model.addVars(nx, name='abs_x_' + str(mark))
		abs_w = gurobi_model.addVars(ny, name='abs_w_' + str(mark))

		gurobi_model.addConstrs((abs_x[i] == gp.abs_(x[i]) for i in range(nx)),
								name='dist_constr_abs_x_step_' + mark)
		gurobi_model.addConstrs((abs_w[i] == gp.abs_(w[i]) for i in range(ny)),
								name='dist_constr_abs_w_step_' + mark)
		gurobi_model.update()

		# Attention: we can use addVar() or addVars(1), but not addVar(1)!
		w_norm = gurobi_model.addVar(name='w_norm_step_' + str(mark))
		x_norm = gurobi_model.addVar(name='x_norm_step_' + str(mark))

		gurobi_model.update()

		if self.norm_type == 'inf':
			gurobi_model.addConstr(x_norm == gp.max_([abs_x[i] for i in range(nx)]),  name='dist_constr_x_norm_step_' + str(mark))
			gurobi_model.addConstr(w_norm == gp.max_([abs_w[i] for i in range(ny)]),  name='dist_constr_w_norm_step_' + str(mark))

		elif self.norm_type == 'one':
			gurobi_model.addConstr(w_norm == abs_w.sum(), name='dist_constr_w_norm_step_' + str(mark))
			gurobi_model.addConstr(x_norm == abs_x.sum(), name='dist_constr_x_norm_step_' + str(mark))
		else:
			raise ValueError('unsupported norm type')

		for i in range(len(self.gamma)):
			gurobi_model.addConstr(w_norm <= self.gamma[i] * x_norm + self.delta[i],  name='dist_constr_norm_bd_step_' + str(i) + '_' + str(mark))

		gurobi_model.update()

		return gurobi_model

class dist_vector:
	def __init__(self, nx, dist_list, input_idx_list, output_idx_list):
		self.dist_list = dist_list
		self.input_idx_list = input_idx_list
		self.output_idx_list = output_idx_list
		self.nx = nx

		# check if there is repeated output
		idx = []
		for item in output_idx_list:
			idx += item

		unique_idx, c = np.unique(idx, return_counts=True)
		self.output_idx_set = unique_idx

		if c.max() > 1:
			raise ValueError('Repeated outputs detected!')

	def random_sample(self, x):
		nx = self.nx
		w = np.zeros(nx)
		num_dist = len(self.dist_list)
		for i in range(num_dist):
			input_idx = self.input_idx_list[i]
			output_idx = self.output_idx_list[i]
			dist = self.dist_list[i]

			x_input = x[input_idx]
			w_sample = dist.random_sample(x_input)
			w[output_idx] = w_sample

		return w

	def add_gurobi_constrs(self, gurobi_model, x_id, w_id, mark):
		num_dist = len(self.dist_list)

		for i in range(num_dist):
			dist = self.dist_list[i]
			input_idx_set = self.input_idx_list[i]
			output_idx_set = self.output_idx_list[i]

			input_ids = [x_id + '[' + str(idx) + ']' for idx in input_idx_set]
			output_ids = [w_id + '[' + str(idx) + ']' for idx in output_idx_set]

			_ = dist.add_gurobi_constrs(gurobi_model, input_ids, output_ids, mark + '_' + str(i))

		return gurobi_model



class nn_vector_field:
	# vector of NNs to represent the nonlinear vector field
	def __init__(self, nx, nn_net_list, input_idx_list, output_idx_list):
		'''examples: nn_net_list = [ nn_net_1, nn_net_2],
			input_idx_list = [ [1, 3], [2, 3]]
			output_idx_list  = [ [2], [3]]

			Input variable x in R^4, output variable y in R^4, and
			y[2] = nn_net_1([x[1],x[3]]), y[3] = nn_net_2([x[2],x[3]])
			y[0] = 0, y[1] = 0 by default
		'''

		self.nn_net_list = nn_net_list
		self.input_idx_list = input_idx_list
		self.output_idx_list = output_idx_list
		self.nx = nx

		'''attention: how to choose domain'''
		self.domain = nn_net_list[0].domain

		# check if there is repeated output
		idx = []
		for item in output_idx_list:
			idx += item

		unique_idx, c = np.unique(idx, return_counts = True)
		self.output_idx_set = unique_idx

		if c.max() > 1:
			raise ValueError('Repeated outputs detected!')

	def evaluate(self, x):
		nx = self.nx
		y = np.zeros(nx)
		num_nn_nets = len(self.nn_net_list)
		for i in range(num_nn_nets):
			input_idx = self.input_idx_list[i]
			output_idx = self.output_idx_list[i]
			nn_net = self.nn_net_list[i]

			input = x[input_idx]
			output = nn_net.evaluate(input)
			y[output_idx] = output

		return y

	def add_gurobi_constrs(self, gurobi_model, input_id, output_id, mark):
		num_nn_nets = len(self.nn_net_list)

		for i in range(num_nn_nets):
			nn_net = self.nn_net_list[i]
			input_idx_set = self.input_idx_list[i]
			output_idx_set = self.output_idx_list[i]

			input_ids = [input_id + '[' + str(idx) + ']' for idx in input_idx_set]
			output_ids = [output_id + '[' + str(idx) + ']' for idx in output_idx_set]

			_ = nn_net.add_gurobi_constrs(gurobi_model, input_ids, output_ids, mark + '_' + str(i))

		return gurobi_model

class approx_dynamics:
	def __init__(self, A, nn_vec, dist_vec):
		self.A = A
		self.nn_vec = nn_vec
		self.dist_vec = dist_vec
		self.nx = A.shape[0]
		self.domain = polyhedral_set(nn_vec.domain.A, nn_vec.domain.b)

	def evaluate(self, x, simu_type = 'nominal'):
		if simu_type == 'nominal':
			y = self.A@x + self.nn_vec.evaluate(x)
		elif simu_type == 'uncertain':
			y = self.A@x + self.nn_vec.evaluate(x)
			w = self.dist_vec.random_sample(x)
			y = y + w
		else:
			raise ValueError('Simulation type not supported')

		return y

	def simulate_traj(self, x, step = 0, simu_type = 'nominal'):
		traj = np.vstack((np.zeros(x.shape), x))
		for i in range(step):
			x_next = self.evaluate(x, simu_type)
			traj = np.vstack((traj, x_next))
			x = x_next

		traj = traj[1:, :]
		return traj

	def add_gurobi_constrs(self, gurobi_model, input_id, output_id, mark, simu_type = 'nominal'):
		'''add the constraint y = \hat{f}(x) while constructing all intermediate variables'''

		nx = self.nx

		x = list()
		for i in range(nx):
			x.append(gurobi_model.getVarByName(input_id + '[' + str(i) + ']'))

		y = list()
		for i in range(nx):
			y.append(gurobi_model.getVarByName(output_id + '[' + str(i) + ']'))

		# add constraint u = Ax
		u = gurobi_model.addVars(nx, lb = -gp.GRB.INFINITY, name = 'linear_output_' + str(mark))
		gurobi_model.addConstrs((u[i] == self.A[i]@x for i in range(self.A.shape[0])), name = 'linear_dynamics_' + str(mark))
		gurobi_model.update()

		# add vector field constraint
		v = gurobi_model.addVars(nx, lb = -gp.GRB.INFINITY, name = 'nn_vec_output_' + str(mark))
		# absent entries from the nn output are set to zero
		unique_output_idx_list = self.nn_vec.output_idx_set
		full_idx_list = [i for i in range(nx)]
		diff_idx_list = list(set(full_idx_list) - set(unique_output_idx_list))
		gurobi_model.addConstrs( (v[idx] == 0 for idx in diff_idx_list), name = 'nn_vec_zero_output_' + str(mark))
		gurobi_model.update()

		# add nn vector field constraints
		_ = self.nn_vec.add_gurobi_constrs(gurobi_model, input_id, 'nn_vec_output_' + str(mark), mark)

		if simu_type == 'uncertain':
			# add disturbance constraint
			w = gurobi_model.addVars(nx, lb = -gp.GRB.INFINITY, name = 'dist_output_' + str(mark))
			dist_unique_output_idx_list = self.dist_vec.output_idx_set
			full_idx_list = [i for i in range(nx)]
			dist_diff_idx_list = list(set(full_idx_list) - set(dist_unique_output_idx_list))
			gurobi_model.addConstrs((w[idx] == 0 for idx in dist_diff_idx_list), name='dist_zero_output_' + str(mark))

			gurobi_model.update()
			_ = self.dist_vec.add_gurobi_constrs(gurobi_model, input_id, 'dist_output_' + str(mark), mark)

			gurobi_model.addConstrs( (y[i] == u[i] + v[i] + w[i] for i in range(nx)), name = 'aggregate_constr_' + str(mark) )
			gurobi_model.update()
		elif simu_type == 'nominal':
			gurobi_model.addConstrs( (y[i] == u[i] + v[i] for i in range(nx)), name = 'aggregate_constr_' + str(mark) )
			gurobi_model.update()
		else:
			raise ValueError('Constraint type not supported')

		return gurobi_model

class polyhedral_set:
	def __init__(self, A, b):
		# polytope defined by Ax <= b
		self.A = A
		self.b = b
		self.X = Polyhedron(A, b)
		self.nx = self.A.shape[1]

	def add_set_constr(self, gurobi_model, input_id, step_id):
		x = list()
		for i in range(self.nx):
			x.append(gurobi_model.getVarByName(input_id + '[' + str(i) + ']'))

		A = self.A
		b = self.b

		gurobi_model.addConstrs( (A[i]@x <= b[i] for i in range(A.shape[0])), name = 'set_constr_step_' + str(step_id))
		gurobi_model.update()
		return gurobi_model

class origin_guard:
	def __init__(self, radius, nx, norm_type = 'inf'):
		self.radius = radius
		self.norm_type = norm_type
		self.nx = nx

	def add_guard_constr(self, gurobi_model, input_id, mark = 'origin_guard'):
		radius = self.radius

		x = list()
		nx = self.nx
		for i in range(nx):
			x.append(gurobi_model.getVarByName(input_id + '[' + str(i) + ']'))

		abs_x = gurobi_model.addVars(nx)

		gurobi_model.addConstrs((abs_x[i] == gp.abs_(x[i]) for i in range(nx)), name='abs_constr_' + str(mark))

		binary_vars = gurobi_model.addVars(nx, name='box_binary_' + str(mark), vtype=gp.GRB.BINARY)

		gurobi_model.addConstrs((abs_x[i] >= radius * binary_vars[i] for i in range(nx)), name='norm_ball_constr_' + str(mark))

		'''attention'''
		gurobi_model.addConstr(binary_vars.sum() == 1, name='norm_ball_binary_constr_' + str(mark))
		# gurobi_model.addConstr(binary_vars.sum() >= 1, name='norm_ball_binary_constr_' + str(mark))

		gurobi_model.update()

		return gurobi_model

class polyhedral_origin_guard:
	def __init__(self, A, b, ref_polytope = None):
		nx = A.shape[1]
		m = A.shape[0]
		self.nx = nx
		self.num_constr = m
		self.A = A
		self.b = b
		self.ref_polytope = ref_polytope
		self.compute_big_M()

	def compute_big_M(self, M = 10):
		ref_polytope = self.ref_polytope
		if ref_polytope is not None:
			lb_list = []
			for i in range(self.num_constr):
				a = self.A[i]
				obj, _ = LP_over_polyhedron(a, ref_polytope)
				lb_list.append(obj - self.b[i])
			self.lb_list = lb_list
		else:
			lb_list = [-M for i in range(self.num_constr)]
			self.lb_list = lb_list
		return lb_list

	def add_guard_constr(self, gurobi_model, input_id, mark = 'origin_guard'):
		x = list()
		nx = self.nx
		for i in range(nx):
			x.append(gurobi_model.getVarByName(input_id + '[' + str(i) + ']'))

		num_constr = self.num_constr
		A = self.A
		b = self.b
		lb_list = self.lb_list
		t = gurobi_model.addVars(num_constr,  name = 'origin_guard_binary_var_' + str(mark), vtype = gp.GRB.BINARY)
		gurobi_model.addConstrs((A[i]@x >= b[i] + lb_list[i]*(1 - t[i]) for i in range(num_constr)), name = 'origin_guard_constr_' + str(mark))
		gurobi_model.addConstr(t.sum() >= 1, name='origin_guard_t_sum_' + str(mark))
		gurobi_model.update()

		return gurobi_model


class Lyapunov_candidate:
	def __init__(self, approx_dyn, order, P):
		self.system = approx_dyn
		self.nx = self.system.nx
		self.order = order
		self.P = P

	def evaluate(self, x):
		traj = self.system.simulate_traj(x, self.order)
		return traj

	def plot_level_set(self, xlim, ylim, level, N_dim = 40):
		# plot the level set of Lyapunov function in 2D
		# N_dim is the number of sample points in each direction

		x_range = np.linspace(xlim[0], xlim[1], N_dim)
		y_range = np.linspace(ylim[0], ylim[1], N_dim)
		xx, yy = np.meshgrid(x_range, y_range)

		zz = np.zeros(xx.shape)
		for i in range(xx.shape[0]):
			for j in range(xx.shape[1]):
				traj = self.evaluate(np.array([xx[i, j], yy[i, j]]))
				zz[i, j] = traj.flatten()@self.P@traj.flatten()


		h = plt.contour(xx, yy, zz, [level], colors='b')
		for i, label in enumerate(h.cvalues):
			h.collections[i].set_label('robust ROA')


class Problem:
	def __init__(self, system, ROI, order, origin_guard, sample_set = None):
		# approximate system
		self.system = system
		# ROI is of the polyhedral_set class
		self.ROI = ROI
		self.order = order
		self.origin_guard = origin_guard
		if sample_set is None:
			sample_set = {'state': [], 'disturbance': []}
		self.sample_set = sample_set

	def set_sample_set(self, samples):
		self.sample_set = samples

	def set_system(self, system):
		self.system = system















