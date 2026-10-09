import pickle
from pathlib import Path
import numpy as np
import os
import utilities.functions as fnc
import utilities.plot_utilities as plt_util
from utilities.parameters import * 

os.environ["CUDA_VISIBLE_DEVICES"] = "" # force JAX to use CPU  # Set JAX to use CPU only, avoiding GPU/TPU usage


# Iterations confing
iters =  [1, 5, 10, 15, 20, 25, 30, 35, 40]

# Path to the models 
dir_alm = Path("Tesi_Repo_Git/single_integrator/saved_models/models_ALM_40_iters")
dir_no_alm = Path("Tesi_Repo_Git/single_integrator/saved_models/models_NO_ALM_40_iters")

# Initilize lists to save the results
to_episodes_alm_list, to_episodes_no_alm_list = [], []
critic_updates_alm_list, critic_updates_no_alm_list = [], []
actor_updates_alm_list, actor_updates_no_alm_list = [], []

mean_costs_alm, std_costs_alm = [], []
mean_costs_no_alm, std_costs_no_alm = [], []


# Start the evaluation loop for each iteration
for i in iters:

  print(f"--- Evaluating iteration {i}/{iters[-1]} ---\n")

  path_alm = dir_alm / f"actor_policy_ALM_iter{i}.pkl"
  path_no_alm = dir_no_alm / f"actor_policy_NO_ALM_iter{i}.pkl"

  # Check if the model files exist
  if not path_alm.exists() or not path_no_alm.exists():
    print(
        f"Model files for iteration {i} do not exist. Skipping this iteration."
    )
    continue

  # Load the actor models
  actor_alm, stats_alm = fnc.load_actor(path_alm)
  actor_no_alm, stats_no_alm = fnc.load_actor(path_no_alm)

  # Exract the training statistics\
  # Extract cumulative TO episodes solved in training
  to_episodes_alm_list.append(stats_alm.get("to_episodes"))
  to_episodes_no_alm_list.append(stats_no_alm.get("to_episodes"))

  # Extract cumulative critic and actor updates in training
  critic_updates_alm_list.append(stats_alm.get("critic_updates", i))
  critic_updates_no_alm_list.append(stats_no_alm.get("critic_updates", i))
  
  actor_updates_alm_list.append(stats_alm.get("actor_updates", i))
  actor_updates_no_alm_list.append(stats_no_alm.get("actor_updates", i))

  # --- Solve the TO problems for the current iteration's actor models ---
  print(f"• Solving TO problems for iteration {i}.\n")

  x_alm, _, flags_alm, costs_alm = fnc.solve_TO_problems(
            actor=actor_alm,
            initial_states=INITIAL_STATES_HARD,
            N=N,
            target=TARGET_POS,
            cost_params=COST_WEIGHTS_ALM,
            obstacle_params=OBSTACLE_PARAMETERS,
            cost_function=fnc.cost_ALM,
            fallback_on_fail=True
        )

  x_no_alm, _, flags_no_alm, costs_no_alm = fnc.solve_TO_problems(
            actor=actor_no_alm,
            initial_states=INITIAL_STATES_HARD,
            N=N,
            target=TARGET_POS,
            cost_params=COST_WEIGHTS_ALM,
            obstacle_params=OBSTACLE_PARAMETERS,
            cost_function=fnc.cost_ALM,
            fallback_on_fail=True
        )

  # Plot for debugg
  plt_util.plot_rollouts_comparison(
    x_alm, x_no_alm, 
    costs_ALM=costs_alm, costs_NO_ALM=costs_no_alm, 
    obstacle_params=OBSTACLE_PARAMETERS, target_pos=TARGET_POS, 
    print_details=True
)

  
  # --- Compute mean and std of costs for the current iteration ---
  mean_costs_alm.append(np.mean(costs_alm))
  std_costs_alm.append(np.std(costs_alm))

  mean_costs_no_alm.append(np.mean(costs_no_alm))
  std_costs_no_alm.append(np.std(costs_no_alm))

  print(f"• Iteration {i} - Mean costs (ALM): {mean_costs_alm[-1]:.2f} ± {std_costs_alm[-1]:.2f}\n")
  print(f"• Iteration {i} - Mean costs (NO ALM): {mean_costs_no_alm[-1]:.2f} ± {std_costs_no_alm[-1]:.2f}\n")


# Conversiont of lists to numpy arrays for plotting
mean_costs_alm = np.array(mean_costs_alm)
std_costs_alm = np.array(std_costs_alm)
mean_costs_no_alm = np.array(mean_costs_no_alm)
std_costs_no_alm = np.array(std_costs_no_alm)


# --- Plot the convergence results using plot_utilities ---
plt_util.plot_cost_convergence(
    iters=iters,
    mean_costs_alm=mean_costs_alm,
    std_costs_alm=std_costs_alm,
    mean_costs_no_alm=mean_costs_no_alm,
    std_costs_no_alm=std_costs_no_alm,
    to_episodes_alm=to_episodes_alm_list,
    to_episodes_no_alm=to_episodes_no_alm_list,
    critic_updates_alm=critic_updates_alm_list,
    actor_updates_alm=actor_updates_alm_list,
    critic_updates_no_alm=critic_updates_no_alm_list,
    actor_updates_no_alm=actor_updates_no_alm_list
)