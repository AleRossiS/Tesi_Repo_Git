RANDOM_SEED = 13  # seed of the randon number generator

# --- PARAMETERS OF THE PROBLEM ---
N = 50      # horizon size
DT = 0.05   # time step
N_X = 2     # State dimension (X, Y)
N_U = 2     # Control dimension (u_x, u_y)
U_MAX = 10  # maximum control input (for both u_x and u_y)


# --- Target, cost and obstacles parameters ---
TARGET_POS = [-7.0, 0.0] 

# ALM Case
COST_WEIGHTS_ALM = {'w_d': 1e2, 'w_p': 5e1, 'w_u': 6e1, 'w_f': 1e1} # w_d distance to target, w_p target valley, w_u control effort, w_f final position cost

# NO ALM Case
COST_WEIGHTS_NO_ALM = {'w_d': 1e2, 'w_p': 5e5, 'w_u': 6e1, 'w_ob': 1e6, 'w_f': 3e2} # distance ot targ, target valley, control effort, obstacles, final pos cost

# Obstacles 
OBSTACLE_PARAMETERS = [
			{'x_ob': 0.0, 'y_ob': 0.0,  'a': 1.5, 'b': 5.0}, # Ellipse 1 (vertical, back)
			{'x_ob': 3, 'y_ob': 4.0,  'a': 4.0, 'b': 1.5}, # Ellipse 2 (horizontal, top)
			{'x_ob': 3, 'y_ob': -4.0, 'a': 4.0, 'b': 1.5}  # Ellipse 3 (horizontal, bottom)
]


#Ellipses tollerance for the feasibility check (to avoid numerical issues)
TOLLERANCE = 1e-4

# --- Initial states ---
INITIAL_STATES = [
	# Out of the C obstacle (Left / Far) - 16 points
	[-12.0,  0.0], [-12.0,  5.0], [-12.0, -5.0], [-12.0, 10.0],
	[-12.0,-10.0], [-10.0,  0.0], [-10.0,  3.0], [-10.0, -3.0],
	[-10.0,  7.5], [-10.0, -7.5], [-8.0,   2.5], [-8.0,  -2.5],
	[-8.0,   6.0], [-8.0,  -6.0], [-8.0,  10.0], [-8.0, -10.0],

	# Inside the C obstacle - 38 points
	[1.8,   0.0], [2.5,   0.0], [3.2,   0.0], [4.0,   0.0],
	[4.5,   0.0], [5.0,   0.0], [1.5,   1.0], [1.5,  -1.0],
	[1.5,   2.0], [1.5,  -2.0], [2.0,   1.2], [2.0,  -1.2],
	[2.0,   2.0], [2.0,  -2.0], [2.5,   0.8], [2.5,  -0.8],
	[2.5,   1.8], [2.5,  -1.8], [3.0,   1.0], [3.0,  -1.0],
	[3.0,   1.8], [3.0,  -1.8], [3.5,   0.8], [3.5,  -0.8],
	[3.5,   1.5], [3.5,  -1.5], [4.0,   0.8], [4.0,  -0.8],
	[4.0,   2.0], [4.0,  -2.0], [4.5,   1.2], [4.5,  -1.2],
	[4.5,   2.3], [4.5,  -2.3], [5.2,   1.2], [5.2,  -1.2],
	[5.5,   2.5], [5.5,  -2.5],

	# Near the boundary (Right / Beyond) - 16 points
	[10.0,  0.0], [13.0,  0.0], [10.0,  4.0], [10.0, -4.0],
	[10.0,  8.0], [10.0, -8.0], [10.0, 10.0], [10.0,-10.0],
	[12.0,  3.0], [12.0, -3.0], [12.0,  7.0], [12.0, -7.0],
	[14.0,  5.0], [14.0, -5.0], [14.0, 10.0], [14.0,-10.0]
]

INITIAL_STATES_HARD = [

	# Inside the C obstacle - 38 points
	[1.8,   0.0], [2.5,   0.0], [3.2,   0.0], [4.0,   0.0],
	[4.5,   0.0], [5.0,   0.0], [1.5,   1.0], [1.5,  -1.0],
	[1.5,   2.0], [1.5,  -2.0], [2.0,   1.2], [2.0,  -1.2],
	[2.0,   2.0], [2.0,  -2.0], [2.5,   0.8], [2.5,  -0.8],
	[2.5,   1.8], [2.5,  -1.8], [3.0,   1.0], [3.0,  -1.0],
	[3.0,   1.8], [3.0,  -1.8], [3.5,   0.8], [3.5,  -0.8],
	[3.5,   1.5], [3.5,  -1.5], [4.0,   0.8], [4.0,  -0.8],
	[4.0,   2.0], [4.0,  -2.0], [4.5,   1.2], [4.5,  -1.2],
	[4.5,   2.3], [4.5,  -2.3], [5.2,   1.2], [5.2,  -1.2],
	[5.5,   2.5], [5.5,  -2.5]

]