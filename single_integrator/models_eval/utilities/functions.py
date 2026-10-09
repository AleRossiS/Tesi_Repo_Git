from .parameters import *
import pickle
from .network import ActorNetwork
import casadi as ca
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from .OCP_solver import OcpSingleIntegrator



def load_actor(filepath):
	"""
	Load the actor model from a file.

	Args:
		filepath (str): The path to the file containing the actor model.
	"""
	# Load the actor model from the specified file
	with open(filepath, 'rb') as f:
		actor_data = pickle.load(f)

	config = actor_data['config']
	network_params = actor_data['params']

	# Initialize the ActorNetwork with the loaded configuration
	actor = ActorNetwork(
		name="Policy_Inference", 
		in_dim=config["in_dim"], 
		out_dim=config["out_dim"], 
		layer_sizes=config["layer_sizes"], 
		u_max=config["u_max"]
	)

	# Set the loaded parameters to the actor
	actor.params = network_params

	#load training statistics 
	stats = actor_data.get("training_stats", {})

	print(f"Actor model loaded from {filepath} with configuration: {config}\n")

	return actor, stats


def elliptical_constraints(x_next, ellips_params):
	"""
	Generate the mathematical constraint equations using CasADi.
	args:
		x_next: The next state of the system (position) for which we want to evaluate the constraints.
		ellips_params: A list of dictionaries, where each dictionary contains the parameters of an ellipse

	returns:
		A list of inequality expressions representing the constraints for each ellipse.  
	"""

	x_pos = x_next[0]
	y_pos = x_next[1]

	constraints = []
	for param in ellips_params:
		x_term = ((x_pos - param['x_ob'])**2) / (param['a']**2)
		y_term = ((y_pos - param['y_ob'])**2) / (param['b']**2)

		g_ellipse = x_term + y_term - 1
		constraints.append(g_ellipse)

	return constraints



def cost_ALM(x, u, target, params = COST_WEIGHTS_ALM):
	"""
	Computes the cost function based on the CACTO paper (excluding the obstacle term l3), thus for the ALM case.
	
	Args:
		x: State vector [x, y].
		u: Control input [vx, vy].
		target: Target coordinates [x_g, y_g].
		params: Dictionary containing weights (w_d, w_p, w_u).
		
	Returns:
		c: The computed scalar cost.
	"""
	
	# Extract weights from the params dictionary
	w_d = params['w_d']      
	w_p = params['w_p']        
	w_u = params['w_u']

	# Extract current position and target coordinates
	x_pos, y_pos = x[0], x[1]
	x_g, y_g = target[0], target[1]
		 
	# Constants from the CACTO paper (i changed them wrt the paper)
	c1 = 1000.0
	c2 = 1000.0
	c3 = 0.1
	c4 = -2 * c3 - 2 * ca.sqrt(c3)
	alpha_1 = 50.0
	
	# --- Compute individual cost terms ---
	
	# l1: Quadratic Tracking Cost (Formula 12)
	l1 = w_d * ((x_pos - x_g)**2 + (y_pos - y_g)**2)
	
	# l2: Target Cost Valley (Formula 13)
	dist_term = ca.sqrt((x_pos - x_g)**2 + c2) + ca.sqrt((y_pos - y_g)**2 + c3) + c4
	l2 = (w_p / alpha_1) * ca.log(ca.exp(-alpha_1 * dist_term) + 1.0)
	
	# l4: Control Effort (Formula 15)
	# Using lib-specific dot product/sumsqr logic
	l4 = w_u * ca.dot(u, u) 
		
	# Total Cost Calculation
	c = (1.0 / c2) * (l1 + l2 + l4 - c1)
	
	return c



def cost_NO_ALM(x, u, target, params = COST_WEIGHTS_NO_ALM):
	"""
	Computes the cost function based on the CACTO paper (excluding the obstacle term l3).
	
	Args:
		x: State vector [x, y].
		u: Control input [vx, vy].
		target: Target coordinates [x_g, y_g].
		params: Dictionary containing weights (w_d, w_p, w_u).
		
	Returns:
		c: The computed scalar cost.
	"""
	
	# Extract weights from the params dictionary
	w_d = params['w_d']      
	w_p = params['w_p']        
	w_u = params['w_u']
	w_ob = params['w_ob']

	# Extract current position and target coordinates
	x_pos, y_pos = x[0], x[1]
	x_g, y_g = target[0], target[1]
	
	# Constants from the CACTO paper (i changed them wrt the paper)
	c1 = 10000.0
	c2 = 100.0
	c3 = 0.1
	c4 = -2 * c3 - 2 * ca.sqrt(c3)
	alpha_1 = 50.0
	alpha_2 = 50.0
	
	# --- Compute individual cost terms ---
	
	# l1: Quadratic Tracking Cost (Formula 12)
	l1 = w_d * ((x_pos - x_g)**2 + (y_pos - y_g)**2)
	
	# l2: Target Cost Valley (Formula 13)
	dist_term = ca.sqrt((x_pos - x_g)**2 + c2) + ca.sqrt((y_pos - y_g)**2 + c3) + c4
	l2 = (w_p / alpha_1) * ca.log(ca.exp(-alpha_1 * dist_term) + 1.0)
	
	# l3: Obstacle Cost (Formula 14)
	l3 = 0
	for obs in OBSTACLE_PARAMETERS:
		
		# Extract obstacle parameters
		xo = obs['x_ob']
		yo = obs['y_ob']
		ao = obs['a'] * 2  # Major axis (full length)
		bo = obs['b'] * 2  # Minor axis (full length)
		# Ellipse: (dx/semi_a)^2 + (dy/semi_b)^2 - 1
		f_ell = ((x_pos - xo) / (ao / 2))**2 + ((y_pos - yo) / (bo / 2))**2 - 1.0
		
		# Log-Sum-Exp term: log(exp(-alpha2 * f_ell) + 1)
		# Clipping the exponent to avoid numerical issues (overflow) 
		exp_term = ca.exp(-alpha_2 * f_ell)
		l3 += (w_ob / alpha_2) * ca.log(exp_term + 1.0) 
	
	# l4: Control Effort (Formula 15)
	l4 = w_u * ca.dot(u, u)

	# Total Cost Calculation
	c = (1.0 / c2) * (l1 + l2 + l4 - c1) 
	
	return c



def dynamic_func(x, u):
	'''Single integrator dynamics.'''
	x_next = x + DT*u
	return x_next



def policy_rollout(actor, x_init_list, N):
	"""
	Perform policy rollouts for a given list of initial states using the provided actor model.

	Args:
		actor: The actor model to use for generating actions.
		x_init_list: A list of initial states of the system.
		N: The number of time steps to rollout.

	Returns:
		all_x_traj: A list containing the rolled-out state trajectories for each initial state.
		all_u_traj: A list containing the rolled-out action trajectories for each initial state.
		all_costs: A list containing the cumulative costs for each trajectory.
	"""
	all_x_traj = []
	all_u_traj = []
	all_costs = []

	# Pure function compiled with JAX JIT to drastically accelerate neural network inference
	@jax.jit
	def fast_action(params, x_in):
		raw_u = actor.model.apply(params, x_in)
		return actor.u_max * jnp.tanh(raw_u)

	# Wrap the initial states loop with tqdm to display a progress bar
	for x_init in tqdm(x_init_list, desc=f"Processing {actor.name}", unit="traj"):
		x_traj = np.zeros((N + 1, N_X))  # Preallocate state trajectory
		u_traj = np.zeros((N, N_U))      # Preallocate action trajectory
		cost_traj = np.zeros(N)            # Preallocate cost trajectory

		# Initialize current state
		x = np.array(x_init)
		x_traj[0] = x

		# Loop through each time step to perform the rollout
		for i in range(N):
			t = i  # Current time step
			x_aug = jnp.array([x[0], x[1], t])  # Augmented input state [x, y, t]
			
			# Evaluate policy using the JIT-compiled fast inference function
			u = fast_action(actor.params, x_aug)
			u_np = np.array(u)
			
			# Propagate system dynamics using physical state [x, y]
			x = dynamic_func(x, u_np)
			
			# Save the updated state and applied control action
			x_traj[i + 1] = np.array(x)
			u_traj[i] = u_np
		# Append the trajectories of the current rollout to the main lists

		cost_traj = compute_trajectory_cost(x_traj, u_traj, cost_fn=cost_ALM, target_pos=TARGET_POS, params=COST_WEIGHTS_ALM)
		all_x_traj.append(x_traj)
		all_u_traj.append(u_traj)
		all_costs.append(cost_traj)
  
	return all_x_traj, all_u_traj, all_costs



def check_trajectory_feasibility(x_traj, obstacle_params):
	"""
	Check if a state trajectory satisfies all elliptical obstacle constraints
	and compute the constraint violation for each point in the trajectory.

	Args:
		x_traj: Trajectory array of shape (N+1, 2).
		obstacle_params: List of dictionaries containing obstacle parameters.

	Returns:
		is_feasible: True if no obstacle is intersected, False otherwise.
		point_violations: NumPy array of shape (N+1,) containing the violation depth at each point.
	"""
	n_points = len(x_traj)
	point_violations = np.zeros(n_points)

	for k, state in enumerate(x_traj):
		x_pos, y_pos = state[0], state[1]
		max_violation_at_k = 0.0

		for obs in obstacle_params:
			# Ellipse equation: (x - x_ob)^2 / a^2 + (y - y_ob)^2 / b^2 - 1 >= 0
			g_val = ((x_pos - obs['x_ob'])**2) / (obs['a']**2) + \
					((y_pos - obs['y_ob'])**2) / (obs['b']**2) - 1.0

			# Constraint violation occurs when g_val < 0
			if g_val < - TOLLERANCE:  # Use a small tolerance to avoid numerical issues
				violation = -g_val
				if violation > max_violation_at_k:
					max_violation_at_k = violation

		point_violations[k] = max_violation_at_k

	# Trajectory is feasible only if all point violations are below a small tolerance to avoid numerical issues
	is_feasible = np.all(point_violations <= TOLLERANCE)  

	# NOTE: the tollerance can be adjusted if needed

	return is_feasible, point_violations


  
def is_inside_obstacle(x_pos, y_pos, obstacle_params=OBSTACLE_PARAMETERS):
	"""
	Check if a given 2D point (x_pos, y_pos) lies inside any of the defined elliptical obstacles.

	Args:
		x_pos (float): X-coordinate of the point.
		y_pos (float): Y-coordinate of the point.
		obstacle_params (list): List of obstacle parameter dictionaries.

	Returns:
		bool: True if the point is strictly inside an obstacle, False otherwise.
	"""
	for obs in obstacle_params:
		g_val = ((x_pos - obs['x_ob'])**2) / (obs['a']**2) + \
				((y_pos - obs['y_ob'])**2) / (obs['b']**2) - 1.0
		if g_val < - TOLLERANCE:  # Use a small tolerance to avoid numerical issues
			return True
	return False


def generate_grid_initial_states(x_bounds=(-15.0, 15.0), y_bounds=(-15.0, 15.0), spacing=1.0, obstacle_params=OBSTACLE_PARAMETERS):
	"""
	Generate a uniform 2D grid of initial states within bounded limits based on step spacing,
	excluding points inside obstacles.

	Args:
		x_bounds (tuple): Min and max limits for X position.
		y_bounds (tuple): Min and max limits for Y position.
		spacing (float): Grid step size (resolution) between neighboring sample points.
		obstacle_params (list): List of obstacle parameter dictionaries.

	Returns:
		x_vals (ndarray): 1D array of sampled X coordinates.
		y_vals (ndarray): 1D array of sampled Y coordinates.
		grid_states (list): Valid initial state positions [x, y] outside obstacles.
	"""
	# Use spacing/2.0 tolerance on upper bound to include the endpoint
	x_vals = np.arange(x_bounds[0], x_bounds[1] + spacing / 2.0, spacing)
	y_vals = np.arange(y_bounds[0], y_bounds[1] + spacing / 2.0, spacing)
	
	grid_states = []
	for y in y_vals:
		for x in x_vals:
			if not is_inside_obstacle(x, y, obstacle_params):
				grid_states.append([x, y])
				
	return x_vals, y_vals, grid_states


def compute_trajectory_cost(x_traj, u_traj, cost_fn, target_pos=TARGET_POS, params=None):
	"""
	Compute the cumulative trajectory cost using the provided stage cost function (e.g., cost_ALM or cost_NO_ALM).
	It includes the final cost term based on the distance to the target as in the OCP.

	Args:
		x_traj (ndarray): State trajectory array of shape (N+1, 2).
		u_traj (ndarray): Control action trajectory array of shape (N, 2).
		cost_fn (callable): Stage cost function (fnc.cost_ALM or fnc.cost_NO_ALM).
		target_pos (list/ndarray): Target coordinates [x_g, y_g].
		params (dict, optional): Weights dictionary for the cost function.

	Returns:
		float: Total accumulated cost over the rollout horizon.
	"""
	total_cost = 0.0
	for k in range(len(u_traj)):
		x_k = x_traj[k]
		u_k = u_traj[k]
		
		# Evaluate stage cost using the passed cost function
		if params is not None:
			c_k = cost_fn(x_k, u_k, target_pos, params)
		else:
			c_k = cost_fn(x_k, u_k, target_pos)
			
		total_cost += float(c_k)

	final_cost = params['w_f'] * np.sum((x_traj[-1] - target_pos)**2) 
	total_cost += final_cost
		
	return total_cost


def evaluate_grid_performance(x_vals, y_vals, grid_states, 
							  x_trajs_alm, u_trajs_alm, 
							  x_trajs_no_alm, u_trajs_no_alm, 
							  obstacle_params=OBSTACLE_PARAMETERS, 
							  target_pos=TARGET_POS):
	"""
	Evaluate trajectory feasibility and cumulative stage costs for all initial states across a 2D grid.

	Args:
		x_vals (ndarray): 1D array of sampled X coordinates.
		y_vals (ndarray): 1D array of sampled Y coordinates.
		grid_states (list): List of initial state positions [x, y].
		x_trajs_alm (list): Rollout state trajectories for ALM actor.
		u_trajs_alm (list): Control input trajectories for ALM actor.
		x_trajs_no_alm (list): Rollout state trajectories for NO ALM actor.
		u_trajs_no_alm (list): Control input trajectories for NO ALM actor.
		obstacle_params (list): List of obstacle parameter dictionaries.
		target_pos (list/ndarray): Target coordinates [x_g, y_g].

	Returns:
		feas_grid (ndarray): 2D array encoding feasibility status across the grid.
		cost_alm_grid (ndarray): 2D array of cumulative ALM rollout costs.
		cost_no_alm_grid (ndarray): 2D array of cumulative NO ALM rollout costs.
	"""
	# Preallocate 2D grid matrices (rows = Y, columns = X)
	feas_grid = np.full((len(y_vals), len(x_vals)), -1, dtype=int)
	cost_alm_grid = np.full((len(y_vals), len(x_vals)), np.nan)
	cost_no_alm_grid = np.full((len(y_vals), len(x_vals)), np.nan)

	# Create coordinate-to-index resolution lookup maps
	x_map = {np.round(val, 4): i for i, val in enumerate(x_vals)}
	y_map = {np.round(val, 4): i for i, val in enumerate(y_vals)}

	for idx, state in enumerate(grid_states):
		col = x_map[np.round(state[0], 4)]
		row = y_map[np.round(state[1], 4)]

		x_alm, u_alm = x_trajs_alm[idx], u_trajs_alm[idx]
		x_no_alm, u_no_alm = x_trajs_no_alm[idx], u_trajs_no_alm[idx]

		# Check feasibility
		feas_alm, _ = check_trajectory_feasibility(x_alm, obstacle_params)
		feas_no_alm, _ = check_trajectory_feasibility(x_no_alm, obstacle_params)

		# Encode status: 0: Both Fail | 1: ALM Fail & NO ALM OK | 2: ALM OK & NO ALM Fail | 3: Both OK
		if feas_alm and feas_no_alm:
			feas_grid[row, col] = 3
		elif feas_alm and not feas_no_alm:
			feas_grid[row, col] = 2
		elif not feas_alm and feas_no_alm:
			feas_grid[row, col] = 1
		else:
			feas_grid[row, col] = 0

		# Calculate costs
		cost_alm_grid[row, col] = compute_trajectory_cost(
			x_alm, u_alm, cost_fn=cost_ALM, target_pos=target_pos, params=COST_WEIGHTS_ALM
		)
		cost_no_alm_grid[row, col] = compute_trajectory_cost(
			x_no_alm, u_no_alm, cost_fn=cost_ALM, target_pos=target_pos, params=COST_WEIGHTS_ALM
		)

		# NOTE: IMPORTANT the same cost function is used for both ALM and NO ALM. This is needed to have a fair comparison of the two policies. One can chose cost_ALM for cost_NO_ALM.

	return feas_grid, cost_alm_grid, cost_no_alm_grid



def solve_TO_problems(actor, initial_states, N, target, cost_params, obstacle_params, cost_function, fallback_on_fail=True):
	"""
	Solve the Trajectory Optimization (TO) problems for a list of initial states. 
	It uses policy rollouts as an initial guess to speed up solver convergence.
	Optionally, it can fallback to a standard solve (no warm-start) if the initial guess fails.

	Args:
		actor: The actor model used to generate initial trajectory guesses.
		initial_states (list): List of initial states [x, y].
		N (int): Time horizon.
		target (list/ndarray): Target position coordinates [x_g, y_g].
		cost_params (dict): Dictionary containing weight parameters for the cost function.
		obstacle_params (list): List of dictionaries containing obstacle parameters.
		cost_function (callable): Stage cost function (fnc.cost_ALM or fnc.cost_NO_ALM).
		fallback_on_fail (bool): If True, retries solving without warm-start when the initial guess fails.

	Returns:
		x_opt_trajs (list): List of optimal state trajectories.
		u_opt_trajs (list): List of optimal control trajectories.
		success_flags (list): List of booleans indicating solver convergence for each initial state.
	"""
	# 1. Generate initial guesses using policy rollouts
	x_trajs_guess, u_trajs_guess, _ = policy_rollout(actor, initial_states, N)

	# 2. Initialize the OCP solver instance
	ocp_solver = OcpSingleIntegrator(N, target, cost_params, obstacle_params, cost_function)

	x_opt_trajs = []
	u_opt_trajs = []
	success_flags = []
	cost_trajs = []

	#3. Solve the OCP for each initial state using the warm-start guesses
	for i, x_init in enumerate(tqdm(initial_states, desc="Solving TO Problems", unit="traj")):
		X_guess = x_trajs_guess[i]
		U_guess = u_trajs_guess[i]


		try: 

			# Try solving with warm-start
			sol = ocp_solver.solve(x_init, N, X_guess=X_guess, U_guess=U_guess)

			# Extract optimal solution trajectories (transposed to match shape (N+1, N_X) and (N, N_U))
			x_opt = np.array(sol.value(ocp_solver.x)).T 
			u_opt = np.array(sol.value(ocp_solver.u)).T

			x_opt_trajs.append(x_opt)
			u_opt_trajs.append(u_opt)
			success_flags.append(True)

		except Exception as e:

			if fallback_on_fail:
			   # Handle cases where IpOpt fails to converge with the warm-start
				try: 
					# Second attempt: fallback to solving without initial guess
					sol = ocp_solver.solve(x_init=x_init, N=N, X_guess=None, U_guess=None)
					
					x_opt = np.array(sol.value(ocp_solver.x)).T 
					u_opt = np.array(sol.value(ocp_solver.u)).T
					
					x_opt_trajs.append(x_opt)
					u_opt_trajs.append(u_opt)
					success_flags.append(True)

				except Exception as fallback_e:
					# Both attempts failed
					x_opt_trajs.append(None)
					u_opt_trajs.append(None)
					success_flags.append(False)

			else:
				# Failed and fallback is disabled
				x_opt_trajs.append(None)
				u_opt_trajs.append(None)
				success_flags.append(False)

		cost = compute_trajectory_cost(x_opt, u_opt, cost_fn=cost_function, target_pos=target, params=cost_params) if success_flags[-1] else None

		cost_trajs.append(cost)



	return x_opt_trajs, u_opt_trajs, success_flags, cost_trajs



