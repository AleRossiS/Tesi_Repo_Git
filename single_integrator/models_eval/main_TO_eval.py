import os
import pickle 
import jax.numpy as jnp
import utilities.functions as fnc
import utilities.plot_utilities as plt_util
from utilities.parameters import * 
import matplotlib.pyplot as plt

"""
Main script for evaluating the TO warm started with the actor policy, for the ALM and NO ALM cases. First, the script solves a set of TO problems for both policies and compares their rollouts (starting from INITIAL_STATES). Then, it generates a grid of initial states and evaluates the performance of both policies on this grid, comparing their costs and feasibility. Finally, it plots the results for visual comparison.
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "" # force JAX to use CPU  # Set JAX to use CPU only, avoiding GPU/TPU usage

# Models parameters paths
filepath_ALM = os.path.join("Tesi_Repo_Git", "single_integrator", "saved_models", "models_ALM_40_iters", "actor_policy_ALM_iter40.pkl")

filepath_NO_ALM = os.path.join("Tesi_Repo_Git", "single_integrator", "saved_models", "models_NO_ALM_40_iters", "actor_policy_NO_ALM_iter40.pkl")

# Create the actors models 
actor_ALM, _ = fnc.load_actor(filepath_ALM)
actor_NO_ALM, _ = fnc.load_actor(filepath_NO_ALM)


## --- Rollout comparison ---

# 1. Solve TO problems ALM case
print("Starting TO problems solving for plot rollout, ALM case...\n")
x_opt_alm, u_opt_alm, status_alm, cost_ALM, _ = fnc.solve_TO_problems(
    actor=actor_ALM,
    initial_states=INITIAL_STATES,
    N=N,
    target=TARGET_POS,
    cost_params=COST_WEIGHTS_ALM,
    obstacle_params=OBSTACLE_PARAMETERS,
    cost_function=fnc.cost_ALM,
    fallback_on_fail=True
)

success_rate_alm = (sum(status_alm) / len(status_alm)) * 100
print(f"Success rate TO (ALM): {success_rate_alm:.2f}% ({sum(status_alm)}/{len(status_alm)})\n")

# 2. Solve TO problems NO ALM case
print("Starting TO problems solving for plot rollout, NO ALM case...\n")
x_opt_no_alm, u_opt_no_alm, status_no_alm, cost_NO_ALM, _ = fnc.solve_TO_problems(
      actor=actor_NO_ALM,
      initial_states=INITIAL_STATES,
      N=N,
      target=TARGET_POS,
      cost_params=COST_WEIGHTS_ALM,
      obstacle_params=OBSTACLE_PARAMETERS,
      cost_function=fnc.cost_ALM,
      fallback_on_fail=True
)

success_rate_no_alm = (sum(status_no_alm) / len(status_no_alm)) * 100
print(f"Success rate TO (NO_ALM): {success_rate_no_alm:.2f}% ({sum(status_no_alm)}/{len(status_no_alm)})\n")

# NOTE: The same cost function is used for both ALM and NO ALM. This is needed to have a fair comparison of the two policies. One can chose cost_ALM for cost_NO_ALM.

plt_util.plot_rollouts_comparison(
    x_trajs_ALM=x_opt_alm,
    x_trajs_NO_ALM=x_opt_no_alm,
    costs_ALM=cost_ALM,
    costs_NO_ALM=cost_NO_ALM,
    obstacle_params=OBSTACLE_PARAMETERS,
    target_pos=TARGET_POS,
)


# --- Heatmap comparison of the two sets of optimized trajectories ---

# 1. Generate 2D grid of initial states (excluding obstacles)
x_vals, y_vals, initial_states = fnc.generate_grid_initial_states(spacing=0.8)

# 2. Solve TO problems ALM case
print("Starting TO problems solving for Costs Comparison - ALM case...\n")
x_opt_alm_grid, u_opt_alm_grid, status_alm_grid, _, _ = fnc.solve_TO_problems(
    actor=actor_ALM,
    initial_states=initial_states,
    N=N,
    target=TARGET_POS,
    cost_params=COST_WEIGHTS_ALM,
    obstacle_params=OBSTACLE_PARAMETERS,
    cost_function=fnc.cost_ALM,
    fallback_on_fail=True
)

success_rate_alm = (sum(status_alm_grid) / len(status_alm_grid)) * 100
print(f"Success rate TO (ALM): {success_rate_alm:.2f}% ({sum(status_alm_grid)}/{len(status_alm_grid)})\n")

# 3. Solve TO problems NO ALM case
print("Starting TO problems solving for Costs Comparison - NO ALM case...\n")
x_opt_no_alm_grid, u_opt_no_alm_grid, status_no_alm_grid, _, _ = fnc.solve_TO_problems(
      actor=actor_NO_ALM,
      initial_states=initial_states,
      N=N,
      target=TARGET_POS,
      cost_params=COST_WEIGHTS_ALM,
      obstacle_params=OBSTACLE_PARAMETERS,
      cost_function=fnc.cost_ALM,
      fallback_on_fail=True
)

success_rate_no_alm = (sum(status_no_alm_grid) / len(status_no_alm_grid)) * 100
print(f"Success rate TO (NO_ALM): {success_rate_no_alm:.2f}% ({sum(status_no_alm_grid)}/{len(status_no_alm_grid)})\n")


# NOTE: The same cost function is used for both ALM and NO ALM. This is needed to have a fair comparison of the two policies. One can chose cost_ALM for cost_NO_ALM.


# Plot the comparison of the two sets of optimized trajectories
feas_grid, cost_alm_grid, cost_no_alm_grid = fnc.evaluate_grid_performance(
        x_vals=x_vals,
        y_vals=y_vals,
        grid_states=initial_states,
        x_trajs_alm=x_opt_alm_grid,
        u_trajs_alm=u_opt_alm_grid,
        x_trajs_no_alm=x_opt_no_alm_grid,
        u_trajs_no_alm=u_opt_no_alm_grid,
        obstacle_params=OBSTACLE_PARAMETERS,
        target_pos=TARGET_POS
    )

plt_util.plot_cost_comparison_heatmap(
        x_vals=x_vals,
        y_vals=y_vals,
        cost_alm_grid=cost_alm_grid,
        cost_no_alm_grid=cost_no_alm_grid,
        feas_grid=feas_grid,
        obstacle_params=OBSTACLE_PARAMETERS,
        target_pos=TARGET_POS,
        only_both_feasible=True, 
        clip_percentile=100.0,
        tol=0.1
    )


# Mostra tutti i plot aperti
plt.show()