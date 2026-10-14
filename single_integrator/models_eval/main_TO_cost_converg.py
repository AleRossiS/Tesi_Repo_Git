import pickle
from pathlib import Path
import numpy as np
import os
import utilities.functions as fnc
import utilities.plot_utilities as plt_util
from utilities.parameters import * 
import matplotlib.pyplot as plt

"""
Main script for evaluating the TO warm started with the actor policy, for the ALM and NO ALM cases. First, the script solves a set of TO problems for both policies and compares their rollouts (starting from INITIAL_STATES) across different actor training iterations. Then  it computes the mean and standard deviation of the costs and times for each policy at each iteration, and plots the convergence results. The evaluation is done on a set of predefined initial states = INITIAL_STATES_HARD.
The following metrics are computed and plotted:
- Mean and standard deviation of costs for each policy at each iteration.
- Mean and standard deviation of times for each policy at each iteration.
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "" # force JAX to use CPU

# Iterations config
iters = [1, 5, 10, 15, 20, 25, 30, 35, 40]

# Path to the models 
dir_alm = Path("Tesi_Repo_Git/single_integrator/saved_models/models_ALM_40_iters")
dir_no_alm = Path("Tesi_Repo_Git/single_integrator/saved_models/models_NO_ALM_40_iters")

# Initialize lists to save the results
to_episodes_alm_list, to_episodes_no_alm_list = [], []
critic_updates_alm_list, critic_updates_no_alm_list = [], []
actor_updates_alm_list, actor_updates_no_alm_list = [], []

mean_costs_alm, q1_costs_alm, q3_costs_alm = [], [], []
mean_costs_no_alm, q1_costs_no_alm, q3_costs_no_alm = [], [], []

mean_times_alm, q1_times_alm, q3_times_alm = [], [], []
mean_times_no_alm, q1_times_no_alm, q3_times_no_alm = [], [], []

rollouts_history = []

# Start the evaluation loop for each iteration
for i in iters:
	print(f"\n================================================")
	print(f"--- Evaluating iteration {i}/{iters[-1]} ---")
	print(f"================================================\n")

	path_alm = dir_alm / f"actor_policy_ALM_iter{i}.pkl"
	path_no_alm = dir_no_alm / f"actor_policy_NO_ALM_iter{i}.pkl"

	if not path_alm.exists() or not path_no_alm.exists():
		print(f"Model files for iteration {i} do not exist. Skipping this iteration.")
		continue

	actor_alm, stats_alm = fnc.load_actor(path_alm)
	actor_no_alm, stats_no_alm = fnc.load_actor(path_no_alm)

	to_episodes_alm_list.append(stats_alm.get("to_episodes"))
	to_episodes_no_alm_list.append(stats_no_alm.get("to_episodes"))

	critic_updates_alm_list.append(stats_alm.get("critic_updates", i))
	critic_updates_no_alm_list.append(stats_no_alm.get("critic_updates", i))
	
	actor_updates_alm_list.append(stats_alm.get("actor_updates", i))
	actor_updates_no_alm_list.append(stats_no_alm.get("actor_updates", i))

	print(f"\n• Solving TO problems for iteration {i}.\n")

	x_alm, _, flags_alm, costs_alm, times_alm = fnc.solve_TO_problems(
			actor=actor_alm,
			initial_states=INITIAL_STATES_HARD,
			N=N,
			target=TARGET_POS,
			cost_params=COST_WEIGHTS_ALM,
			obstacle_params=OBSTACLE_PARAMETERS,
			cost_function=fnc.cost_ALM,
			fallback_on_fail=True
		)

	x_no_alm, _, flags_no_alm, costs_no_alm, times_no_alm = fnc.solve_TO_problems(
			actor=actor_no_alm,
			initial_states=INITIAL_STATES_HARD,
			N=N,
			target=TARGET_POS,
			cost_params=COST_WEIGHTS_ALM,
			obstacle_params=OBSTACLE_PARAMETERS,
			cost_function=fnc.cost_ALM,
			fallback_on_fail=True
		)

	rollouts_history.append({
		'iter': i,
		'x_alm': x_alm,
		'x_no_alm': x_no_alm,
		'costs_alm': costs_alm,
		'costs_no_alm': costs_no_alm
	})

	# --- Compute mean and quartiles (Q1, Q3) for the current iteration ---
	mean_costs_alm.append(np.mean(costs_alm))
	q1_costs_alm.append(np.percentile(costs_alm, 25))
	q3_costs_alm.append(np.percentile(costs_alm, 75))

	mean_costs_no_alm.append(np.mean(costs_no_alm))
	q1_costs_no_alm.append(np.percentile(costs_no_alm, 25))
	q3_costs_no_alm.append(np.percentile(costs_no_alm, 75))

	mean_times_alm.append(np.mean(times_alm))
	q1_times_alm.append(np.percentile(times_alm, 25))
	q3_times_alm.append(np.percentile(times_alm, 75))

	mean_times_no_alm.append(np.mean(times_no_alm))
	q1_times_no_alm.append(np.percentile(times_no_alm, 25))
	q3_times_no_alm.append(np.percentile(times_no_alm, 75))

	print(f"• Iteration {i} - Mean costs (ALM): {mean_costs_alm[-1]:.2f} (Q1: {q1_costs_alm[-1]:.2f}, Q3: {q3_costs_alm[-1]:.2f})\n")
	print(f"• Iteration {i} - Mean costs (NO ALM): {mean_costs_no_alm[-1]:.2f} (Q1: {q1_costs_no_alm[-1]:.2f}, Q3: {q3_costs_no_alm[-1]:.2f})\n")
	print(f"• Iteration {i} - Mean times (ALM): {mean_times_alm[-1]:.4f} (Q1: {q1_times_alm[-1]:.4f}, Q3: {q3_times_alm[-1]:.4f})\n")
	print(f"• Iteration {i} - Mean times (NO ALM): {mean_times_no_alm[-1]:.4f} (Q1: {q1_times_no_alm[-1]:.4f}, Q3: {q3_times_no_alm[-1]:.4f})\n\n")


plt_util.plot_all_iterations_rollouts(
    rollouts_history=rollouts_history,
    obstacle_params=OBSTACLE_PARAMETERS,
    target_pos=TARGET_POS
)

# Conversion of lists to numpy arrays for plotting
mean_costs_alm = np.array(mean_costs_alm)
q1_costs_alm = np.array(q1_costs_alm)
q3_costs_alm = np.array(q3_costs_alm)

mean_costs_no_alm = np.array(mean_costs_no_alm)
q1_costs_no_alm = np.array(q1_costs_no_alm)
q3_costs_no_alm = np.array(q3_costs_no_alm)

mean_times_alm = np.array(mean_times_alm)       
q1_times_alm = np.array(q1_times_alm)
q3_times_alm = np.array(q3_times_alm)

mean_times_no_alm = np.array(mean_times_no_alm)
q1_times_no_alm = np.array(q1_times_no_alm)
q3_times_no_alm = np.array(q3_times_no_alm)


# --- Plot the convergence results using plot_utilities ---
plt_util.plot_cost_convergence(
    iters=iters,
    mean_costs_alm=mean_costs_alm,
    q1_costs_alm=q1_costs_alm,
    q3_costs_alm=q3_costs_alm,
    mean_costs_no_alm=mean_costs_no_alm,
    q1_costs_no_alm=q1_costs_no_alm,
    q3_costs_no_alm=q3_costs_no_alm,
    to_episodes_alm=to_episodes_alm_list,
    to_episodes_no_alm=to_episodes_no_alm_list,
    mean_times_alm=mean_times_alm,        
    q1_times_alm=q1_times_alm,
    q3_times_alm=q3_times_alm,
    mean_times_no_alm=mean_times_no_alm,   
    q1_times_no_alm=q1_times_no_alm,
    q3_times_no_alm=q3_times_no_alm
)

plt.show()  # Show all plots at the end of the evaluation loop


# NOTE: in the plot_cost_convergence function I commented the fill_between for the mean times plots, since it was not very informative and made the plot less clear.