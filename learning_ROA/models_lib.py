# save models

import numpy as np

from types import SimpleNamespace
from math import fmod


class Dynamics_to_fit:
	def __init__(self, dyn_fcn, A, dt = None):
		self.dyn_fcn = dyn_fcn

		self.A = A
		self.nx = A.shape[0]
		self.dt = dt

		if self.dt is None:
			self.type = 'discrete'
		else:
			self.type = 'continuous'

	def err_dynamics(self, x):
		if self.type == 'discrete':
			err = self.dyn_fcn(x) - self.A@x
		elif self.type == 'continuous':
			err = x + self.dt * self.dyn_fcn(x) - self.A @ x

		return err

	def simulate_traj(self, x, N_step = 5):
		traj = np.vstack((np.zeros(x.shape), x))
		for i in range(N_step):
			try:
				if self.type == 'discrete':
					x_next = self.dyn_fcn(x)
				elif self.type == 'continuous':
					x_next = x + self.dt*self.dyn_fcn(x)
			except Exception as e:
				break

			# stop simulating diverging trajectories
			'''attention'''
			if np.linalg.norm(x_next, np.inf) > 1e3:
				break

			traj = np.vstack((traj, x_next))
			x = x_next

		traj = traj[1:,:]
		return traj

	def dynamics(self, x):
		if self.dt is None:
			return self.dyn_fcn(x)
		else:
			return x + self.dt*self.dyn_fcn(x)


####################################################################################
# rational dynamics
#####################################################################################

def rational_dyn(x):
	# rational dynamical system adapted from
	# D. Coutinho and C. E. de Souza, “Local stability analysis and domain of attraction estimation for
	# a class of uncertain nonlinear discrete-time systems,” International Journal of Robust and Nonlinear
	# Control, 2013.

	x_1 = x[0] - (x[0] + x[1] ** 3) / (1 + x[1] ** 2)
	x_2 = x[1] + (x[0] ** 3 - 0.25 * x[1]) / (1 + x[1] ** 2)
	return np.array([x_1, x_2])


####################################################################################
# switched 3D system
#####################################################################################

def polynomial_3D(x):
	# this is a continuous time system from
	# A. I. Doban and M. Lazar, “Computation of Lyapunov functions for nonlinear differential equations via
	# a massera-type construction,” IEEE Transactions on Automatic Control,	2017.

	x_1 = x[0] * (x[0] ** 2 + x[1] ** 2 - 1) - x[1] * (x[2] ** 2 + 1)
	x_2 = x[1] * (x[0] ** 2 + x[1] ** 2 - 1) + x[0] * (x[2] ** 2 + 1)
	x_3 = 10 * x[2] * (x[2] ** 2 - 1)
	return np.array([x_1, x_2, x_3])


###############################################################################################
# predator-prey model
###############################################################################################

def predator_prey(x):
	x_1 = 0.5 * x[0] - x[0] * x[1]
	x_2 = -0.5 * x[1] + x[0] * x[1]

	y = np.array([x_1, x_2])
	return y

def predator_prey_coordinate_1(x):
	x_1 = -x[0]*x[1]
	return np.array([x_1])

def predator_prey_coordinate_2(x):
	x_2 = x[0]*x[1]
	return np.array([x_2])

###############################################################################################
# van der pol system
###############################################################################################

def van_der_pol(x):
	"""
 	continuous time Van der Pol system in reverse time: dx_1/dt = -x_2, dx_2/dt = x_1 - x_2 + x_1^2*x_2
    """
	x_1 = 0.9988 * x[0] - 0.0488 * x[1] - 0.0012 * x[0] ** 2 * x[1]
	x_2 = 0.0488 * x[0] + 0.9500 * x[1] + 0.0488 * x[0] ** 2 * x[1]
	
	y = np.array([x_1, x_2])
	return y
