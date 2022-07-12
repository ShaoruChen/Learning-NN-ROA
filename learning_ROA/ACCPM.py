
import sys
sys.path.append("..\pympc")

import gurobipy as gp
import numpy as np
import cvxpy as cp
from .utilities import pickle_file
from pympc.geometry.polyhedron import Polyhedron


class gurobi_options:
	def __init__(self, time_limit = 1e-9, best_obj = 1e-4, MIPFocus = None):
		self.time_limit = time_limit
		self.best_obj = best_obj
		self.MIPFocus = MIPFocus

class ACCPM_options:
	def __init__(self, max_iter = 30, tol = 1e-4):
		self.max_iter = max_iter
		self.tol = tol


class Learner:
	def __init__(self, problem):
		self.order = problem.order
		self.sample_set = problem.sample_set
		self.system = problem.system
		self.nx = problem.system.nx

		self.ACCPM_infeas = 0

		system = self.system
		self.gamma_var = 0.0

		# state_samples: list of 2 X nx numpy array
		if self.sample_set is None:
			self.x_0_samples = None
			self.y_0_samples = None
			self.num_samples = 0
		else:
			state_samples = self.sample_set['state']
			num_samples = len(state_samples)
			self.num_samples = num_samples

			x_0_samples = []
			for i in range(num_samples):
				x_0 = state_samples[i][0]
				pred_traj = system.simulate_traj(x_0, self.order, 'nominal')
				x_0_samples.append(pred_traj)

			y_0_samples = []
			for i in range(num_samples):
				y_0 = state_samples[i][1]
				pred_traj = system.simulate_traj(y_0, self.order, 'nominal')
				y_0_samples.append(pred_traj)

			self.x_0_samples = x_0_samples
			self.y_0_samples = y_0_samples

		# initialize
		self.feas_status = None
		self.feas_solver_time = None
		self.ac_status = None
		self.ac = None
		self.ac_solver_time = None


	def feasibility_test(self, options = None):
		if options is None:
			alpha = 0.0
			beta = 1.0
		else:
			alpha = options.alpha
			beta = options.beta

		order = self.order
		sample_set = self.sample_set
		system = self.system
		nx = self.nx

		num_samples = self.num_samples

		print('CVXPY started... \n')
		P_var = cp.Variable(((order+1)*nx,(order+1)*nx), symmetric = True)
		gamma_var = cp.Variable(1)

		print('CVXPY constructing constraints... \n')
		constr = []
		if num_samples > 0:
			x_0_samples = self.x_0_samples
			y_0_samples = self.y_0_samples

			norm_list = [np.linalg.norm(y_0_samples[i].reshape(-1,1)@y_0_samples[i].reshape(1,-1) -
							x_0_samples[i].reshape(-1, 1)@x_0_samples[i].reshape(1,-1),'fro') for i in range(num_samples)]

			constr += [(cp.quad_form(y_0_samples[i].flatten(), P_var) - cp.quad_form(x_0_samples[i].flatten(), P_var))/norm_list[i] <= 0 for i in range(num_samples)]

		constr += [P_var >> alpha*np.eye((order+1)*nx)]
		constr += [P_var << beta*np.eye((order+1)*nx)]
		constr += [P_var >> gamma_var*np.eye((order+1)*nx)]
		constr += [gamma_var >= 0]

		prob = cp.Problem(cp.Minimize(-gamma_var), constr)
		prob.solve(solver = cp.MOSEK, verbose = True)
		status = prob.status

		if gamma_var.value <= 1e-11:
			status = 'pathological'

		solver_time = prob.solver_stats.solve_time

		self.feas_status = status
		self.feas_solver_time = solver_time
		self.gamma_var = gamma_var.value

		return status

	def analytic_center(self, options = None):

		order = self.order
		sample_set = self.sample_set
		system = self.system
		nx = self.nx

		num_samples = self.num_samples

		print('CVXPY started... \n')
		P_var = cp.Variable(((order+1)*nx,(order+1)*nx), symmetric = True)

		print('CVXPY constructing constraints... \n')
		constr = []
		obj = 0

		if num_samples > 0:
			x_0_samples = self.x_0_samples
			y_0_samples = self.y_0_samples

			norm_list = [np.linalg.norm(y_0_samples[i].reshape(-1,1)@y_0_samples[i].reshape(1,-1) -
						 x_0_samples[i].reshape(-1,1)@x_0_samples[i].reshape(1,-1),'fro') for i in range(num_samples)]

			obj += sum([-cp.log((cp.quad_form(x_0_samples[i].flatten(), P_var) - cp.quad_form(y_0_samples[i].flatten(), P_var))/norm_list[i]) for i in range(num_samples)])

		obj += -cp.log_det(P_var)
		obj += -cp.log_det(np.eye((order+1)*nx) - P_var)

		prob = cp.Problem(cp.Minimize(obj), constr)
		prob.solve(solver = cp.MOSEK, verbose = True)

		P_sol = P_var.value
		status = prob.status
		solver_time = prob.solver_stats.solve_time

		self.ac_status = status
		self.ac = P_sol
		self.ac_solver_time = solver_time

		return status, P_sol

	def find_analytic_center(self):
		feas_status = self.feasibility_test()
		if feas_status in ['infeasible', 'unbounded', 'pathological']:
			self.ACCPM_infeas = 1
			return None, self.ACCPM_infeas
		else:
			try:
				ac_status, ac_P = self.analytic_center()
			except Exception as e:
				print('Analytic center solving problem', e)
				self.ACCPM_infeas = 1
				ac_P = None

			return ac_P, self.ACCPM_infeas


class Verifier:
	def __init__(self, problem, P, gurobi_model):
		self.order = problem.order
		self.ROI = problem.ROI
		self.system = problem.system
		self.nx = problem.system.nx
		self.P = P
		self.gurobi_model = gurobi_model

	def verify_candidate(self):
		order = self.order
		nx = self.nx
		P = self.P

		gurobi_model = self.gurobi_model

		# extract relevant variables
		x = list()
		y = list()
		for i in range(order+1):
			for j in range(nx):
				x.append(gurobi_model.getVarByName('x_' + str(i) + '[' + str(j) + ']'))

			for j in range(nx):
				y.append(gurobi_model.getVarByName('y_' + str(i) + '[' + str(j) + ']'))

		expr_y = y@P@y
		expr_x = x@P@x

		obj = expr_y - expr_x

		gurobi_model.setObjective(obj, gp.GRB.MAXIMIZE)
		gurobi_model.optimize()

		# extract solutions
		solver_time = gurobi_model.Runtime
		gurobi_status = gurobi_model.Status

		if gurobi_status not in [2, 15]:
			sol = {'obj': None, 'status': gurobi_status, 'x_sol': None, 'y_sol': None, 'w_sol': None, 'solver_time': solver_time}
			return sol

		obj_value = gurobi_model.objVal

		x_opt = list()
		for i in range(nx):
			x_opt.append(x[i].X)
		x_opt = np.array(x_opt)

		y_opt = list()
		for i in range(nx):
			y_opt.append(y[i].X)

		y_opt = np.array(y_opt)

		w_opt = list()

		w = []
		for j in range(nx):
			w.append(gurobi_model.getVarByName('dist_output_0' + '[' + str(j) + ']'))

		for i in range(nx):
			w_opt.append(w[i].X)
		w_opt = np.array(w_opt)

		self.MIP_obj = obj_value
		self.MIP_status = gurobi_status
		self.MIP_x = x_opt
		self.MIP_y = y_opt
		self.MIP_w = w_opt

		sol = {'obj': obj_value, 'status': gurobi_status, 'x_sol': x_opt, 'y_sol': y_opt, 'w_sol': w_opt, 'solver_time': solver_time}

		return sol

class ACCPM_algorithm:
	def __init__(self, problem, alg_options = None, solver_options = None):
		# result = 'feasible', 'infeasible', or 'undecided'

		if alg_options is None:
			alg_options = ACCPM_options()

		if solver_options is None:
			solver_options = gurobi_options()

		self.result = None
		self.alg_options = alg_options
		self.solver_options = solver_options
		self.gamma_values = []

		self.problem = problem
		self.max_iter = alg_options.max_iter
		self.tol = alg_options.tol

		self.candidate_record = []
		self.MIP_value_record = []
		self.MIP_x_record = []
		self.MIP_y_record = []
		self.MIP_w_record = []
		self.num_iter = 0

		self.verifier_solver_time = []
		self.learner_solver_time = []

		self.init_gurobi_model()


	def reset_problem(self, problem):
		self.problem = problem

	def init_gurobi_model(self):
		gurobi_model = gurobi_model_base_construction(self.problem, self.solver_options)
		self.gurobi_model = gurobi_model
		return gurobi_model

	def ACCPM_iter(self):
		# one iteration of the cutting plane method
		problem = self.problem
		alg_status = 'undecided'

		learner = Learner(problem)
		candidate_P, infeas_flag = learner.find_analytic_center()
		self.candidate_record.append(candidate_P)
		self.learner_solver_time.append([learner.feas_solver_time, learner.ac_solver_time])
		self.gamma_values.append(learner.gamma_var)
		'''attention'''
		del learner

		if infeas_flag == 1:
			alg_status = 'infeasible'
			return alg_status

		verif = Verifier(problem, candidate_P, self.gurobi_model)
		verifier_sol = verif.verify_candidate()
		gurobi_status = verifier_sol['status']
		if gurobi_status not in [2, 15]:
			alg_status = 'infeasible_verifier'
			return alg_status

		obj_value = verifier_sol['obj']

		'''attention'''
		del verif

		self.MIP_value_record.append(obj_value)
		self.MIP_x_record.append(verifier_sol['x_sol'])
		self.MIP_y_record.append(verifier_sol['y_sol'])
		self.MIP_w_record.append(verifier_sol['w_sol'])

		MIP_solver_time = verifier_sol['solver_time']
		self.verifier_solver_time.append(MIP_solver_time)

		# expand the sample set
		counterexamples = np.vstack((verifier_sol['x_sol'], verifier_sol['y_sol']))
		problem.sample_set['state'].append(counterexamples)
		problem.sample_set['disturbance'].append(verifier_sol['w_sol'])
		self.problem = problem

		if obj_value < -self.tol:
			alg_status = 'feasible'

		return alg_status


	def ACCPM_main_algorithm(self):
		alg_status = 'undecided'
		max_iter = self.max_iter

		for i in range(max_iter):
			print(f'iter {i} out of {max_iter} iterations')
			iter_status = self.ACCPM_iter()
			self.num_iter += 1
			if iter_status in ['infeasible', 'feasible', 'infeasible_verifier']:
				alg_status = iter_status
				self.result = alg_status
				self.save_data()
				return alg_status

		self.result = alg_status
		self.save_data()
		return alg_status

	def save_data(self, file_name = 'ACCPM_running_log'):
		data_to_save = {'result': self.result, 'alg_options': self.alg_options, 'solver_options': self.solver_options,
						'max_iter': self.max_iter, 'tol':self.tol, 'candidate_record':self.candidate_record,
						'MIP_value_record': self.MIP_value_record, 'MIP_x_record': self.MIP_x_record, 'MIP_y_record': self.MIP_y_record, 'gamma_values': self.gamma_values,
						'MIP_w_record': self.MIP_w_record, 'num_iter': self.num_iter, 'MIP_solver_time': self.verifier_solver_time, 'learner_solver_time': self.learner_solver_time}
		pickle_file(data_to_save, file_name)


def gurobi_model_init(name = 'stability', options = None):
	gurobi_model = gp.Model(name)
	gurobi_model.Params.FeasibilityTol = 1e-9
	gurobi_model.Params.IntFeasTol = 1e-9
	gurobi_model.Params.OptimalityTol = 1e-9

	gurobi_model.Params.NonConvex = 2

	if options is not None:
		if options.time_limit > 1e-3:
			gurobi_model.setParam("TimeLimit", options.time_limit)

		if options.MIPFocus is not None:
			'''attention'''
			gurobi_model.Params.MIPFocus = options.MIPFocus

		gurobi_model.Params.BestObjStop = options.best_obj

	gurobi_model.update()

	return gurobi_model

def gurobi_model_addVars(gurobi_model, var_dict):
	"""
	Add continuous variables in the gurobi_model.
	"""
	for name, dim in var_dict.items():
		gurobi_model.addVars(dim, lb = -gp.GRB.INFINITY, name = name)

	gurobi_model.update()
	return gurobi_model


def gurobi_model_base_construction(problem, options):
	system = problem.system
	ROI = problem.ROI
	origin_guard = problem.origin_guard
	order = problem.order

	gurobi_model = gurobi_model_init(name='base_model', options=options)
	nx = system.nx

	var_dict = {'x_0': nx, 'y_0': nx}
	for i in range(order):
		var_dict['x_' + str(i + 1)] = nx
		var_dict['y_' + str(i + 1)] = nx
	gurobi_model = gurobi_model_addVars(gurobi_model, var_dict)

	_ = system.add_gurobi_constrs(gurobi_model,  'x_0', 'y_0', '0', 'uncertain')

	for i in range(order):
		_ = system.add_gurobi_constrs(gurobi_model, 'x_' + str(i), 'x_' + str(i+1), 'step_x_' + str(i), 'nominal')
		_ = system.add_gurobi_constrs(gurobi_model, 'y_' + str(i), 'y_' + str(i+1), 'step_y_' + str(i), 'nominal')

	# add domain constraint
	_ = ROI.add_set_constr(gurobi_model, 'x_0', 'ROI')

	'''attention: add redundante set constraints'''
	# # add redundant constraints
	# domain = system.domain
	# for i in range(order):
	# 	_ = domain.add_set_constr(gurobi_model, 'x_' + str(i), 'domain_x_' + str(i))
	# 	_ = domain.add_set_constr(gurobi_model, 'y_' + str(i), 'domain_y_' + str(i))

	# add origin guard constraint
	# if origin_guard.radius > 1e-6:
	# 	_ = origin_guard.add_guard_constr(gurobi_model, 'x_0', 'guard')

	if origin_guard is not None:
		_ = origin_guard.add_guard_constr(gurobi_model, 'x_0', 'guard')

	gurobi_model.update()
	return gurobi_model



def find_covering_box(problem, options, order = None):
	gurobi_model = gurobi_model_base_construction(problem, options)
	nx = problem.system.nx
	if order is None:
		order = problem.order

	y = list()
	for i in range(nx):
		y.append(gurobi_model.getVarByName('y_' + str(order) + '[' + str(i) + ']'))

	x = list()
	for i in range(nx):
		x.append(gurobi_model.getVarByName('x_' + str(order) + '[' + str(i) + ']'))

	x_lbs = []
	x_ubs = []
	for i in range(nx):
		c = np.zeros(nx)
		c[i] = 1.0
		obj = c@x
		gurobi_model.setObjective(obj, gp.GRB.MAXIMIZE)
		gurobi_model.optimize()
		x_ubs.append(gurobi_model.objVal)

	for i in range(nx):
		c = np.zeros(nx)
		c[i] = 1.0
		obj = c@x
		gurobi_model.setObjective(obj, gp.GRB.MINIMIZE)
		gurobi_model.optimize()
		x_lbs.append(gurobi_model.objVal)

	y_lbs = []
	y_ubs = []
	for i in range(nx):
		c = np.zeros(nx)
		c[i] = 1.0
		obj = c@y
		gurobi_model.setObjective(obj, gp.GRB.MAXIMIZE)
		gurobi_model.optimize()
		y_ubs.append(gurobi_model.objVal)

	for i in range(nx):
		c = np.zeros(nx)
		c[i] = 1.0
		obj = c@y
		gurobi_model.setObjective(obj, gp.GRB.MINIMIZE)
		gurobi_model.optimize()
		y_lbs.append(gurobi_model.objVal)

	lbs = [min(x_lbs[i], y_lbs[i]) for i in range(nx)]
	ubs = [max(x_ubs[i], y_ubs[i]) for i in range(nx)]
	return np.array(lbs), np.array(ubs)
















