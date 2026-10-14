import pickle
from pathlib import Path
import numpy as np
import os
import utilities.functions as fnc
import utilities.plot_utilities as plt_util
from utilities.parameters import * 
import matplotlib.pyplot as plt

"""
This script evaluates the performance of two policies (with and without ALM) over a series of actor trainig iterations. It loads the saved models, performs rollouts, computes violation metrics, and visualizes the results. The evaluation is done on a set of predefined initial states and obstacle parameters.
The following metrics are computed and plotted:
- Rolout trajectories of bot policies for each iteration from initial states.
- Mean and standard deviation of constraint violations for each policy for a set of initial states.
- Total number of violating points for each policy at each iteration.

The initial states used are INITIAL_STATES_HARD.
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "" # force JAX to use CPU

# Iterations config
iters = [1, 5, 10, 15, 20, 25, 30, 35, 40]

# Path to the models 
dir_alm = Path("Tesi_Repo_Git/single_integrator/saved_models/models_ALM_40_iters")
dir_no_alm = Path("Tesi_Repo_Git/single_integrator/saved_models/models_NO_ALM_40_iters")

# Initialize lists to save the results
to_episodes_alm_list, to_episodes_no_alm_list = [], []

mean_viols_alm, std_viols_alm, num_viols_alm = [], [], []
mean_viols_no_alm, std_viols_no_alm, num_viols_no_alm = [], [], []

rollouts_history = []


# Start the evaluation loop for each iteration
for i in iters:
    print(f"\n================================================")
    print(f"Evaluating policy rollouts for iteration {i}/{iters[-1]}")
    print(f"=================================================\n")

    path_alm = dir_alm / f"actor_policy_ALM_iter{i}.pkl"
    path_no_alm = dir_no_alm / f"actor_policy_NO_ALM_iter{i}.pkl"

    if not path_alm.exists() or not path_no_alm.exists():
        print(f"Model files for iteration {i} do not exist. Skipping this iteration.")
        continue

    # Load the actor models
    actor_alm, stats_alm = fnc.load_actor(path_alm)
    actor_no_alm, stats_no_alm = fnc.load_actor(path_no_alm)

    # Extract cumulative TO episodes solved in training
    to_episodes_alm_list.append(stats_alm.get("to_episodes", 0))
    to_episodes_no_alm_list.append(stats_no_alm.get("to_episodes", 0))

    # --- Perform pure policy rollouts ---
    print(f"\n• Rolling out policy for iteration {i}.\n")

    x_alm, u_alm, costs_alm = fnc.policy_rollout(actor_alm, INITIAL_STATES_HARD, N)
    x_no_alm, u_no_alm, costs_no_alm = fnc.policy_rollout(actor_no_alm, INITIAL_STATES_HARD, N)

    # --- Compute Violation Metrics directly inline ---
    viols_alm = []
    num_v_alm = 0
    for x_traj in x_alm:
        feas, p_viols = fnc.check_trajectory_feasibility(x_traj, OBSTACLE_PARAMETERS)
        viols_alm.extend(p_viols)  
        num_v_alm += np.sum(p_viols > 0.0)  

    mean_v_alm, std_v_alm = np.mean(viols_alm), np.std(viols_alm)

    viols_no_alm = []
    num_v_no = 0
    for x_traj in x_no_alm:
        feas, p_viols = fnc.check_trajectory_feasibility(x_traj, OBSTACLE_PARAMETERS)
        viols_no_alm.extend(p_viols)  
        num_v_no += np.sum(p_viols > 0.0)  

    mean_v_no, std_v_no = np.mean(viols_no_alm), np.std(viols_no_alm)

    # Save metrics
    mean_viols_alm.append(mean_v_alm)
    std_viols_alm.append(std_v_alm)
    num_viols_alm.append(num_v_alm)

    mean_viols_no_alm.append(mean_v_no)
    std_viols_no_alm.append(std_v_no)
    num_viols_no_alm.append(num_v_no)

    print(f"\n• Iteration {i} - Mean Violations (ALM): {mean_v_alm:.4f} ± {std_v_alm:.4f} | Total Violating Points: {num_v_alm}")
    print(f"• Iteration {i} - Mean Violations (NO ALM): {mean_v_no:.4f} ± {std_v_no:.4f} | Total Violating Points: {num_v_no}\n\n")

    # --- Append current rollouts to history ---
    rollouts_history.append({
        'iter': i,
        'x_alm': x_alm,
        'x_no_alm': x_no_alm,
        'costs_alm': costs_alm,
        'costs_no_alm': costs_no_alm
    })


# --- Plot all rollouts in a single dynamic grid window ---
plt_util.plot_all_iterations_rollouts(
    rollouts_history=rollouts_history,
    obstacle_params=OBSTACLE_PARAMETERS,
    target_pos=TARGET_POS,
    show_cost=False,      
    show_violations=True  
)

# Conversion of lists to numpy arrays for plotting
mean_viols_alm = np.array(mean_viols_alm)
std_viols_alm = np.array(std_viols_alm)
num_viols_alm = np.array(num_viols_alm)

mean_viols_no_alm = np.array(mean_viols_no_alm)
std_viols_no_alm = np.array(std_viols_no_alm)
num_viols_no_alm = np.array(num_viols_no_alm)

to_episodes_alm_list = np.array(to_episodes_alm_list)
to_episodes_no_alm_list = np.array(to_episodes_no_alm_list)


# --- Plot the convergence results of the violations ---
plt_util.plot_violation_convergence(
    iters=iters,
    mean_viol_alm=mean_viols_alm,
    std_viol_alm=std_viols_alm,
    num_viol_alm=num_viols_alm,
    mean_viol_no_alm=mean_viols_no_alm,
    std_viol_no_alm=std_viols_no_alm,
    num_viol_no_alm=num_viols_no_alm
)

plt.show()  # Show all plots at the end of the evaluation loop