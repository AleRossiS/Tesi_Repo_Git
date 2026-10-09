import os
import pickle 
import jax.numpy as jnp
import utilities.functions as fnc
import utilities.plot_utilities as plt_util
from utilities.parameters import * 
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "" # force JAX to use CPU  # Set JAX to use CPU only, avoiding GPU/TPU usage

# Models parameters paths
# Models parameters paths
filepath_ALM = os.path.join("Tesi_Repo_Git", "single_integrator", "saved_models", "models_ALM_40_iters", "actor_policy_ALM_iter40.pkl")

filepath_NO_ALM = os.path.join("Tesi_Repo_Git", "single_integrator", "saved_models", "models_NO_ALM_40_iters", "actor_policy_NO_ALM_iter40.pkl")

# Create the actors models 
actor_ALM, _ = fnc.load_actor(filepath_ALM)
actor_NO_ALM, _ = fnc.load_actor(filepath_NO_ALM)

# Do policy rollouts for both ALM and NO ALM cases (unpacking state and action trajectories)
x_trajs_ALM, u_trajs_ALM, costs_ALM = fnc.policy_rollout(actor_ALM, INITIAL_STATES, N)
x_trajs_NO_ALM, u_trajs_NO_ALM, costs_NO_ALM = fnc.policy_rollout(actor_NO_ALM, INITIAL_STATES, N)

# # Plot the comparison of the two sets of rolled-out trajectories
plt_util.plot_rollouts_comparison(x_trajs_ALM, x_trajs_NO_ALM, costs_ALM, costs_NO_ALM, OBSTACLE_PARAMETERS, TARGET_POS)


# --- Grid comparison ---

# 1. Generate 2D grid of initial states (excluding obstacles)
x_vals, y_vals, grid_states = fnc.generate_grid_initial_states(spacing=0.8)

# 2. Perform policy rollouts
x_trajs_ALM_grid, u_trajs_ALM_grid, _  = fnc.policy_rollout(actor_ALM, grid_states, N)
x_trajs_NO_ALM_grid, u_trajs_NO_ALM_grid, _ = fnc.policy_rollout(actor_NO_ALM, grid_states, N)

# 3. Evaluate feasibility and trajectory costs across the grid
feas_grid, cost_alm_grid, cost_no_alm_grid = fnc.evaluate_grid_performance(
    x_vals, y_vals, grid_states, 
    x_trajs_ALM_grid, u_trajs_ALM_grid, 
    x_trajs_NO_ALM_grid, u_trajs_NO_ALM_grid
)

# 4. Plot heatmaps
plt_util.plot_feasibility_heatmap(x_vals, y_vals, feas_grid)
plt_util.plot_cost_comparison_heatmap(x_vals, y_vals, cost_alm_grid, cost_no_alm_grid, feas_grid, only_both_feasible=True)

print("\n")



plt.show()