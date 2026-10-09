import casadi as ca
import jax.numpy as jnp
from .parameters import *
from . import functions as fnc

class OcpSingleIntegrator:
	'''
	A class to solve an Optimal Control Problem (OCP) for a single integrator.
	'''

	def __init__(self, N, target, cost_params, obstacle_params, cost_function):

		self.N = N
		self.target = target
		self.cost_params = cost_params
		self.obstacle_params = obstacle_params
		self.cost_function = cost_function

	def solve(self, x_init, N, X_guess=None, U_guess=None):
		self.opti = ca.Opti() # Use casadi\
		
		# create the state and control variables
		self.x = self.opti.variable(N_X, N+1) 
		self.u = self.opti.variable(N_U, N)
		x = self.x
		u = self.u

		# warm-start the solver with the provided initial guess (if any)
		if(X_guess is not None):
			for i in range(N+1):
				self.opti.set_initial(x[:, i], X_guess[i,:])
		else:
			# otherwise initialize the state trajectory with the initial state
			for i in range(N+1):
				self.opti.set_initial(x[:, i], x_init)
		if(U_guess is not None):
			for i in range(N):
				self.opti.set_initial(u[:, i], U_guess[i,:])
		else:
			# otherwise initialize the control at 0
			for i in range(N):
				self.opti.set_initial(u[:, i], 0.0) 
				
		# define the cost function
		self.cost = 0
		self.running_costs = [None,]*(N+1) # Preallocate a list
		self.final_cost = None
		
		for i in range(N+1):
			# Running cost  
			if(i<N):
				self.running_costs[i] = self.cost_function(x[:, i], u[:, i], self.target, self.cost_params)
				self.cost += self.running_costs[i]
			# Final Cost (simple quadratic cost on the final state)
			else:
				self.final_cost = ca.sumsqr(x[:, N] - self.target) # Simple quadratic final cost
				self.cost += self.cost_params['w_f'] * self.final_cost

			
			# Case ALM i impose constraints on the obstacles, for NO ALM i add the obstacle cost to the cost function
			if(self.cost_function == fnc.cost_ALM):
				obs_constraints = fnc.elliptical_constraints(x[:, i], self.obstacle_params)
				for g in obs_constraints:
					self.opti.subject_to(g >= 0)


		# Define the dynamics constraints and teh controls constraints 
		for i in range(N):            
			
			self.opti.subject_to( x[:, i+1]== fnc.dynamic_func(x[:, i], u[:, i]) )
			self.opti.subject_to( self.opti.bounded(-U_MAX, u[:, i], U_MAX) ) # Control limits

		# Set the initial state
		self.opti.subject_to(x[:, 0]==x_init)
		
		# Set final state constraints
		#self.opti.subject_to(x[:, N] == self.target)
		# NOTE: if the final consrt is used you get failed trajectories (not wanted)
		
		self.opti.minimize(self.cost)

		# Solve the problem with IpOpt
		opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
		self.opti.solver("ipopt", opts)

		return self.opti.solve()
