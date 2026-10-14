import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
import matplotlib.colors as mcolors
from . import functions as fnc
from .parameters import *
import math



def plot_rollouts_comparison(x_trajs_ALM, x_trajs_NO_ALM, 
                             costs_ALM=None, costs_NO_ALM=None, 
                             obstacle_params=OBSTACLE_PARAMETERS, target_pos=TARGET_POS, 
                             print_details=True):
    """
    Compares and visualizes rollouts (Policy or TO) for ALM and NO-ALM policies.

    Plots state trajectories, obstacles, and target positions side-by-side while checking 
    feasibility against obstacles, handling optional costs (ignoring NaN values), and 
    printing detailed performance metrics to the console.

    Args:
        x_trajs_ALM (list of np.ndarray): List of state trajectories for ALM.
        x_trajs_NO_ALM (list of np.ndarray): List of state trajectories for NO-ALM.
        costs_ALM (np.ndarray, optional): Cost grid or array for ALM. Defaults to None.
        costs_NO_ALM (np.ndarray, optional): Cost grid or array for NO-ALM. Defaults to None.
        obstacle_params (list of dict, optional): Elliptical obstacles parameters.
        target_pos (array-like, optional): Target coordinates [x, y].
        print_details (bool, optional): Whether to print details to the console.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    datasets = [
        ("ALM Policy Rollouts", x_trajs_ALM, costs_ALM, ax1),
        ("NO ALM Policy Rollouts", x_trajs_NO_ALM, costs_NO_ALM, ax2)
    ]
    
    n_total = len(x_trajs_ALM)
    summary_stats = {}

    # Check if cost data is available for both cases
    has_costs = (costs_ALM is not None) and (costs_NO_ALM is not None)

    # Flatten cost matrices if provided
    costs_ALM_flat = np.array(costs_ALM).flatten() if costs_ALM is not None else None
    costs_NO_ALM_flat = np.array(costs_NO_ALM).flatten() if costs_NO_ALM is not None else None

    # --- 1. Console Detailed Output ---
    if print_details:
        print("\n" + "="*85)
        print(f"{'POLICY ROLLOUT VIOLATION COMPARISON (ALM vs NO ALM)':^85}")
        print("="*85)

        for idx in range(n_total):
            x_init = x_trajs_ALM[idx][0]
            
            f_alm, v_alm = fnc.check_trajectory_feasibility(x_trajs_ALM[idx], obstacle_params)
            f_no, v_no = fnc.check_trajectory_feasibility(x_trajs_NO_ALM[idx], obstacle_params)
            
            max_v_alm = np.max(v_alm)
            max_v_no = np.max(v_no)

            status_alm = "OK (Feasible)" if f_alm else f"VIOL (Max: {max_v_alm:.4f})"
            status_no = "OK (Feasible)" if f_no else f"VIOL (Max: {max_v_no:.4f})"

            if has_costs:
                c_alm_val = costs_ALM_flat[idx]
                c_no_val = costs_NO_ALM_flat[idx]

                # Handle NaN values gracefully in individual printout
                c_alm_str = f"{c_alm_val:.2f}" if not np.isnan(c_alm_val) else "N/A"
                c_no_str = f"{c_no_val:.2f}" if not np.isnan(c_no_val) else "N/A"

                print(
                    f"Rollout {idx+1:02d} | Start: ({float(x_init[0]):5.1f},"
                    f" {float(x_init[1]):5.1f}) | ALM: {status_alm:22s}"
                    f" {c_alm_str} | NO ALM: {status_no:22s}"
                    f" {c_no_str}"
                )
            else:
                print(
                    f"Rollout {idx+1:02d} | Start: ({float(x_init[0]):5.1f},"
                    f" {float(x_init[1]):5.1f}) | ALM: {status_alm:22s} | NO ALM: {status_no:22s}"
                )

    # --- 2. Plot Trajectories ---
    for title, x_trajs, costs, ax in datasets:
        # Draw obstacles
        for obs in obstacle_params:
            ellipse = Ellipse(
                xy=(obs['x_ob'], obs['y_ob']),
                width=obs['a'] * 2, height=obs['b'] * 2,
                edgecolor='green', facecolor='green', alpha=1, zorder=1
            )
            ax.add_patch(ellipse)

        # Draw target position
        ax.plot(target_pos[0], target_pos[1], marker='*', color='red', 
                markeredgecolor='none', markersize=16, label='Target', zorder=6)

        n_feasible, n_violating = 0, 0
        maxv_list = []

        for i in range(n_total):
            x_traj = x_trajs[i]
            x_init = x_traj[0]

            feasible, point_violations = fnc.check_trajectory_feasibility(x_traj, obstacle_params)
            maxv = np.max(point_violations)
            
            if feasible:
                n_feasible += 1
            else:
                n_violating += 1
            maxv_list.append(maxv)

            line_color = 'blue' if feasible else 'orange'

            # Plot trajectory line
            ax.plot(x_traj[:, 0], x_traj[:, 1], color=line_color, alpha=0.7, linewidth=1.2, zorder=2)
            
            # Identify and plot points inside/outside obstacles
            inside_mask = np.array([fnc.is_inside_obstacle(pt[0], pt[1], obstacle_params) for pt in x_traj])
            
            if np.any(~inside_mask):
                ax.plot(x_traj[~inside_mask, 0], x_traj[~inside_mask, 1], 'o', 
                        color=line_color, markersize=3, alpha=0.6, zorder=3)
            
            if np.any(inside_mask):
                ax.plot(x_traj[inside_mask, 0], x_traj[inside_mask, 1], 'o', 
                        color='red', markersize=4, zorder=5)

            # Plot initial state
            ax.plot(x_init[0], x_init[1], 'ko', markersize=3, zorder=4)

        avg_max_viol = np.mean(maxv_list)

        # Compute average cost ignoring NaN values
        if costs is not None:
            avg_cost = np.nanmean(costs)
        else:
            avg_cost = None

        summary_stats[title] = (n_feasible, n_violating, avg_max_viol, avg_cost)

        # Display summary text box on plot
        box_text = f"Feasible: {n_feasible}/{n_total}\nViolating: {n_violating}/{n_total}\nAvg Max Viol: {avg_max_viol:.4f}"
        if avg_cost is not None and not np.isnan(avg_cost):
            box_text += f"\nAvg Cost: {avg_cost:.2f}"

        ax.text(0.5, 0.02, box_text,
                transform=ax.transAxes, fontsize=10, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=1.0),
                zorder=100)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("X [m]", fontsize=12)
        ax.set_ylabel("Y [m]", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')

    # --- 3. Final Console Summary ---
    if print_details:
        print("-" * 85)
        for title, (n_feas, n_viol, avg_v, avg_c) in summary_stats.items():
            tag = "ALM Summary   " if "ALM" in title and "NO ALM" not in title else "NO ALM Summary"
            cost_str = f" | Avg Cost: {avg_c:.2f}" if (avg_c is not None and not np.isnan(avg_c)) else ""
            print(f"{tag} -> Feasible: {n_feas}/{n_total} | "
                  f"Violating: {n_viol}/{n_total} | "
                  f"Avg Max Viol: {avg_v:.4f}{cost_str}")
        print("=" * 85 + "\n")

    # Figure legend
    legend_elements = [
        plt.Line2D([0], [0], color='green', marker='o', label='Feasible Trajectory', markersize=5),
        plt.Line2D([0], [0], color='orange', marker='o', label='Unfeasible Trajectory', markersize=5),
        plt.Line2D([0], [0], color='none', marker='o', markerfacecolor='red', markeredgecolor='red', label='Point Inside Obstacle', markersize=6),
        plt.Line2D([0], [0], color='none', marker='o', markerfacecolor='black', markeredgecolor='black', label='Start Point', markersize=5),
        Patch(facecolor='green', alpha=0.35, edgecolor='darkgreen', label='Obstacles'),
        plt.Line2D([0], [0], marker='*', color='red', label='Target', linestyle='None', markersize=12)
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=6, fontsize=9.5, framealpha=0.9)
    plt.tight_layout()
    plt.show()




def plot_all_iterations_rollouts(rollouts_history, obstacle_params=OBSTACLE_PARAMETERS, target_pos=TARGET_POS, show_cost=True, show_violations=True):
    """
    Plots all iteration rollouts in a single window containing a dynamic grid of subfigures.
    Each subfigure corresponds to a single iteration and contains two side-by-side plots 
    (ALM vs NO ALM) for direct comparison.

    Args:
        rollouts_history (list of dict): List containing rollout data for each iteration.
        obstacle_params (list of dict, optional): Elliptical obstacles parameters.
        target_pos (array-like, optional): Target coordinates [x, y].
        show_cost (bool, optional): Whether to display average cost in the summary box.
        show_violations (bool, optional): Whether to display average max violation in the summary box.
    """
    n_iters = len(rollouts_history)
    if n_iters == 0:
        return

    # --- 1. Figure and Grid Setup ---
    cols = int(math.ceil(math.sqrt(n_iters)))
    rows = int(math.ceil(n_iters / cols))

    # Create the main constrained figure
    fig = plt.figure(figsize=(cols * 7, rows * 4), layout='constrained')
    fig.suptitle("Evolution of Policy Rollouts Across Iterations", fontsize=20, fontweight='bold')

    # Generate a grid of independent subfigures with increased spacing for better separation
    subfigs = fig.subfigures(rows, cols, wspace=0.1, hspace=0.15).flatten()

    # --- 2. Iterate and Plot Subfigures ---
    for idx in range(rows * cols):
        subfig = subfigs[idx]
        
        # Skip excess subfigures if the grid is larger than the number of iterations
        if idx >= n_iters:
            continue
            
        # --- Modifica: Separazione più netta tramite bordo e sfondo della subfigure ---
        subfig.patch.set_facecolor('#f8f9fa')  # Sfondo leggermente grigio
        subfig.patch.set_edgecolor('gray')     # Colore del bordo
        subfig.patch.set_linewidth(1.5)        # Spessore del bordo

        item = rollouts_history[idx]
        iter_num = item['iter']
        
        # Set the title for the current iteration subfigure
        subfig.suptitle(f"ITERATION {iter_num}", fontweight='bold', fontsize=13, color='darkblue')
        
        # Create 2 side-by-side plots INSIDE the current subfigure
        axs = subfig.subplots(1, 2)
        
        datasets = [
            ("ALM", item['x_alm'], item['costs_alm'], axs[0]),
            ("NO ALM", item['x_no_alm'], item['costs_no_alm'], axs[1])
        ]
        
        # --- 3. Inline Plotting Logic (ALM vs NO ALM) ---
        for title, x_trajs, costs, ax in datasets:
            # Draw obstacles
            for obs in obstacle_params:
                ellipse = Ellipse(
                    xy=(obs['x_ob'], obs['y_ob']),
                    width=obs['a'] * 2, height=obs['b'] * 2,
                    edgecolor='green', facecolor='green', alpha=1, zorder=1
                )
                ax.add_patch(ellipse)

            # Draw target position
            ax.plot(target_pos[0], target_pos[1], marker='*', color='red', 
                    markeredgecolor='none', markersize=12, label='Target', zorder=6)

            n_total = len(x_trajs)
            n_feasible, n_violating = 0, 0
            maxv_list = []

            for i in range(n_total):
                x_traj = x_trajs[i]
                x_init = x_traj[0]

                feasible, point_violations = fnc.check_trajectory_feasibility(x_traj, obstacle_params)
                maxv = np.max(point_violations)
                
                if feasible:
                    n_feasible += 1
                else:
                    n_violating += 1
                maxv_list.append(maxv)

                line_color = 'blue' if feasible else 'orange'

                # Plot trajectory
                ax.plot(x_traj[:, 0], x_traj[:, 1], color=line_color, alpha=0.6, linewidth=1.0, zorder=2)

                # Check for point violations and plot red dots if inside obstacles
                inside_mask = np.array([fnc.is_inside_obstacle(pt[0], pt[1], obstacle_params) for pt in x_traj])
                if np.any(inside_mask):
                    ax.plot(x_traj[inside_mask, 0], x_traj[inside_mask, 1], 'o', 
                            color='red', markersize=2.5, zorder=5)

                # Plot initial state
                ax.plot(x_init[0], x_init[1], 'ko', markersize=2, zorder=4)


            # Create a summary box with feasibility and cost information
            box_text_lines = []
            
            if show_violations:
                box_text_lines.append(f"Feas: {n_feasible}/{n_total} | Viol: {n_violating}/{n_total}")
                
            if show_cost:
                avg_cost = np.nanmean(costs) if costs is not None else None
                if avg_cost is not None and not np.isnan(avg_cost):
                    box_text_lines.append(f"Avg Cost: {avg_cost:.2f}")

            # Show the summary box only if there are lines to display
            if box_text_lines:
                box_text = "\n".join(box_text_lines)
                ax.text(0.5, 0.02, box_text,
                        transform=ax.transAxes, fontsize=9, ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                        zorder=100)

            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xticks([]) # Remove axis ticks to keep the grid clean
            ax.set_yticks([])
            ax.set_aspect('equal', adjustable='box')
            
            # Add a subtle separation border
            for spine in ax.spines.values():
                spine.set_color('lightgray')
                spine.set_linewidth(0.5)

    # --- 4. Global Legend ---
    legend_elements = [
        plt.Line2D([0], [0], color='blue', alpha=0.6, label='Feasible Traj'),
        plt.Line2D([0], [0], color='orange', alpha=0.6, label='Unfeasible Traj'),
        plt.Line2D([0], [0], color='none', marker='o', markerfacecolor='red', markeredgecolor='none', label='Point Inside Obstacle', markersize=6),
        Patch(facecolor='green', alpha=1, label='Obstacles'),
        plt.Line2D([0], [0], marker='*', color='red', label='Target', linestyle='None', markersize=10)
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=11, bbox_to_anchor=(0.5, -0.03))
    #plt.show()






def plot_feasibility_heatmap(x_vals, y_vals, feas_grid, obstacle_params=OBSTACLE_PARAMETERS, target_pos=TARGET_POS):
    """
    Pointwise feasibility comparison heatmap with smoothly blended background regions and crisp overlaid symbols:
    - Navy star ('*'): ALM Feasible & NO ALM Fail (value = 2)
    - Solid white circle ('o'): Both Feasible (value = 3)
    - Dark red cross ('x'): ALM Fail & NO ALM Feasible (value = 1)
    - Dark orange square ('s'): Both Fail (value = 0)
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    color_map_rgba = {
        0: mcolors.to_rgba('peachpuff', alpha=0.8),
        1: mcolors.to_rgba('lightpink', alpha=0.8),
        2: mcolors.to_rgba('lightblue', alpha=0.8),
        3: mcolors.to_rgba('gainsboro', alpha=0.8)
    }

    H, W = feas_grid.shape
    rgba_grid = np.zeros((H, W, 4))

    for val, rgba in color_map_rgba.items():
        mask = (feas_grid == val)
        rgba_grid[mask] = rgba

    rgba_grid[feas_grid == -1] = [0.0, 0.0, 0.0, 0.0]

    ax.imshow(
        rgba_grid,
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        origin='lower',
        interpolation='bicubic',
        zorder=1
    )

    X, Y = np.meshgrid(x_vals, y_vals)
    plot_mask = (feas_grid != -1)
    valid_x = X[plot_mask]
    valid_y = Y[plot_mask]
    valid_vals = feas_grid[plot_mask]

    mask_both_fail = (valid_vals == 0)
    mask_no_alm_only = (valid_vals == 1)
    mask_alm_only = (valid_vals == 2)
    mask_both_feas = (valid_vals == 3)

    ax.scatter(valid_x[mask_both_fail], valid_y[mask_both_fail], 
               marker='s', c='darkorange', edgecolor='none', s=16, alpha=0.9, zorder=3)
    
    ax.scatter(valid_x[mask_no_alm_only], valid_y[mask_no_alm_only], 
               marker='x', c='darkred', linewidths=1.2, s=16, alpha=0.9, zorder=3)
    
    ax.scatter(valid_x[mask_alm_only], valid_y[mask_alm_only], 
               marker='*', c='navy', edgecolor='none', s=22, alpha=0.9, zorder=3)
    
    ax.scatter(valid_x[mask_both_feas], valid_y[mask_both_feas], 
               marker='o', c='white', edgecolor='none', s=14, alpha=0.95, zorder=3)

    for idx, obs in enumerate(obstacle_params):
        ellipse = Ellipse(
            xy=(obs['x_ob'], obs['y_ob']),
            width=obs['a'] * 2, height=obs['b'] * 2,
            edgecolor='none', facecolor='green', alpha=1.0, zorder=4
        )
        ax.add_patch(ellipse)

    target_line = ax.plot(target_pos[0], target_pos[1], marker='*', color='red', 
                          markeredgecolor='none', markersize=18, label='Target', zorder=5)[0]

    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', label='ALM Feasible & NO ALM Fail', 
                   markerfacecolor='navy', markersize=9),
        plt.Line2D([0], [0], marker='o', color='w', label='Both Feasible', 
                   markerfacecolor='white', markeredgecolor='black', markersize=7),
        plt.Line2D([0], [0], marker='x', color='w', label='ALM Fail & NO ALM Feasible', 
                   markeredgecolor='darkred', markersize=7, markeredgewidth=1.2),
        plt.Line2D([0], [0], marker='s', color='w', label='Both Fail', 
                   markerfacecolor='darkorange', markeredgecolor='none', markersize=7),
        Patch(facecolor='green', edgecolor='none', label='Obstacles'),
        target_line
    ]

    fig.suptitle("Feasibility Map Comparison (ALM vs NO ALM)", fontsize=14, fontweight='bold', y=0.98)
    
    # Legend
    fig.legend(
        handles=legend_elements, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.02), 
        bbox_transform=ax.transAxes, 
        ncol=6, 
        fontsize=8.5, 
        framealpha=0.9
    )

    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)
    ax.set_xlim([x_vals[0], x_vals[-1]])
    ax.set_ylim([y_vals[0], y_vals[-1]])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    #plt.show()


def plot_cost_comparison_heatmap(x_vals, y_vals, cost_alm_grid, cost_no_alm_grid, feas_grid, 
                                 obstacle_params=OBSTACLE_PARAMETERS, target_pos=TARGET_POS,
                                 only_both_feasible=True, clip_percentile=98.0, tol=0.5):
    """
    Pointwise cost comparison heatmap with smooth RGBA bicubic interpolation.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    cost_diff = np.full_like(cost_alm_grid, dtype=float, fill_value=np.nan)
    
    if only_both_feasible:
        plot_mask = (feas_grid == 3)
    else:
        plot_mask = (feas_grid != -1)
        
    cost_diff[plot_mask] = cost_alm_grid[plot_mask] - cost_no_alm_grid[plot_mask]

    X, Y = np.meshgrid(x_vals, y_vals)
    valid_diffs = cost_diff[~np.isnan(cost_diff)]

    if len(valid_diffs) > 0:
        if clip_percentile is not None and clip_percentile < 100.0:
            max_val = np.percentile(np.abs(valid_diffs), clip_percentile)
        else:
            max_val = np.max(np.abs(valid_diffs))
            
        max_val = max(max_val, 1.0)
    else:
        max_val = 1.0

    cmap = plt.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)

    H, W = cost_diff.shape
    rgba_grid = np.zeros((H, W, 4))
    infeasible_rgba = mcolors.to_rgba('whitesmoke', alpha=0.6)

    for r in range(H):
        for c in range(W):
            if feas_grid[r, c] == -1:
                rgba_grid[r, c] = [0.0, 0.0, 0.0, 0.0]
            elif np.isnan(cost_diff[r, c]):
                rgba_grid[r, c] = infeasible_rgba
            else:
                color = cmap(norm(cost_diff[r, c]))
                rgba_grid[r, c] = [color[0], color[1], color[2], 0.85]

    ax.imshow(
        rgba_grid,
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        origin='lower',
        interpolation='bicubic',
        zorder=1
    )

    valid_x = X[plot_mask]
    valid_y = Y[plot_mask]
    valid_vals = cost_diff[plot_mask]

    mask_lower = valid_vals < -tol
    mask_equal = np.abs(valid_vals) <= tol
    mask_higher = valid_vals > tol

    ax.scatter(valid_x[mask_lower], valid_y[mask_lower], 
               marker='*', c='navy', s=22, alpha=0.9, zorder=3)
    
    ax.scatter(valid_x[mask_equal], valid_y[mask_equal], 
               marker='o', c='white', edgecolor='none', s=14, alpha=0.95, zorder=3)
    
    ax.scatter(valid_x[mask_higher], valid_y[mask_higher], 
               marker='x', c='darkred', linewidths=1.2, s=16, alpha=0.9, zorder=3)

    # Colorbar Reale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cost Difference ($C_{ALM} - C_{NO\\_ALM}$)", fontsize=11)

    for idx, obs in enumerate(obstacle_params):
        ellipse = Ellipse(
            xy=(obs['x_ob'], obs['y_ob']),
            width=obs['a'] * 2, height=obs['b'] * 2,
            edgecolor='none', facecolor='green', alpha=1.0, zorder=4
        )
        ax.add_patch(ellipse)

    target_line = ax.plot(target_pos[0], target_pos[1], marker='*', color='red', 
                          markeredgecolor='none', markersize=18, label='Target', zorder=5)[0]

    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', label='ALM Lower Cost ($C_{ALM} < C_{NO\\_ALM}$)', 
                   markerfacecolor='navy', markeredgecolor='navy', markersize=9),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Equal Cost ($|\\Delta C| \\leq {tol}$)', 
                   markerfacecolor='white', markeredgecolor='black', markersize=6),
        plt.Line2D([0], [0], marker='x', color='w', label='NO ALM Lower Cost ($C_{ALM} > C_{NO\\_ALM}$)', 
                   markeredgecolor='darkred', markersize=7, markeredgewidth=1.2),
        Patch(facecolor='green', edgecolor='none', label='Obstacles'),
        target_line
    ]

    title_suffix = "(Both Feasible Only)" if only_both_feasible else "(All Valid States)"
    fig.suptitle(f"Cost Difference Map {title_suffix}", fontsize=14, fontweight='bold', y=0.94)
    
    # Legend
    fig.legend(
        handles=legend_elements, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.02), 
        bbox_transform=ax.transAxes, 
        ncol=5, 
        fontsize=8.5, 
        framealpha=0.9
    )

    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)
    ax.set_xlim([x_vals[0], x_vals[-1]])
    ax.set_ylim([y_vals[0], y_vals[-1]])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    #plt.show()



def plot_cost_convergence(
    iters,
    mean_costs_alm,
    std_costs_alm,
    mean_costs_no_alm,
    std_costs_no_alm,
    to_episodes_alm=None,
    to_episodes_no_alm=None,
    mean_times_alm=None,    
    std_times_alm=None,
    mean_times_no_alm=None,
    std_times_no_alm=None   
):
    """
    Plots the convergence of the mean TO trajectory cost, TO episodes solved, 
    and mean OCP resolution time (with +-1 std dev bands) against training iterations.

    Args:
        iters (list or np.ndarray): Training iterations.
        mean_costs_alm (list or np.ndarray): Mean costs for ALM policy.
        std_costs_alm (list or np.ndarray): Standard deviations of costs for ALM policy.
        mean_costs_no_alm (list or np.ndarray): Mean costs for NO ALM policy.
        std_costs_no_alm (list or np.ndarray): Standard deviations of costs for NO ALM policy.
        to_episodes_alm (list or np.ndarray, optional): Cumulative TO episodes solved for ALM policy. Defaults to None.
        to_episodes_no_alm (list or np.ndarray, optional): Cumulative TO episodes solved for NO ALM policy. Defaults to None.
        mean_times_alm (list or np.ndarray, optional): Mean OCP resolution times for ALM policy. Defaults to None.
        std_times_alm (list or np.ndarray, optional): Standard deviations of resolution times for ALM policy. Defaults to None.
        mean_times_no_alm (list or np.ndarray, optional): Mean OCP resolution times for NO ALM policy. Defaults to None.
        std_times_no_alm (list or np.ndarray, optional): Standard deviations of resolution times for NO ALM policy. Defaults to None.
    """
    # Create 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # --- PLOT 1: Mean Cost vs Training Iterations ---
    ax1.plot(iters, mean_costs_alm, "o-", color="blue", label="ALM Policy + TO Warm-Start", linewidth=2)
    ax1.fill_between(iters, mean_costs_alm - std_costs_alm, mean_costs_alm + std_costs_alm, color="blue", alpha=0.15)
    
    ax1.plot(iters, mean_costs_no_alm, "s--", color="orange", label="NO ALM Policy + TO Warm-Start", linewidth=2)
    ax1.fill_between(iters, mean_costs_no_alm - std_costs_no_alm, mean_costs_no_alm + std_costs_no_alm, color="orange", alpha=0.15)

    ax1.set_title("Mean TO Cost vs Training Iterations", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Iterations", fontsize=11)
    ax1.set_ylabel("Mean Trajectory Cost", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # --- PLOT 2: Mean Cost vs TO Episodes Solved in Training ---
    if to_episodes_alm is not None and len(to_episodes_alm) > 0:
        ax2.plot(to_episodes_alm, mean_costs_alm, "o-", color="blue", label="ALM", linewidth=2)
        ax2.fill_between(to_episodes_alm, mean_costs_alm - std_costs_alm, mean_costs_alm + std_costs_alm, color="blue", alpha=0.15)

        ax2.plot(to_episodes_no_alm, mean_costs_no_alm, "s--", color="orange", label="NO ALM", linewidth=2)
        ax2.fill_between(to_episodes_no_alm, mean_costs_no_alm - std_costs_no_alm, mean_costs_no_alm + std_costs_no_alm, color="orange", alpha=0.15)

    ax2.set_title("Mean TO Cost vs TO Episodes Solved", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cumulative TO Episodes Solved in Training", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    # --- PLOT 3: Mean OCP Resolution Time vs Training Iterations ---
    if mean_times_alm is not None and mean_times_no_alm is not None:
        ax3.plot(iters, mean_times_alm, "o-", color="blue", label="ALM", linewidth=2)
        if std_times_alm is not None:
            ax3.fill_between(iters, mean_times_alm - std_times_alm, mean_times_alm + std_times_alm, color="blue", alpha=0.15)

        ax3.plot(iters, mean_times_no_alm, "s--", color="orange", label="NO ALM", linewidth=2)
        if std_times_no_alm is not None:
            ax3.fill_between(iters, mean_times_no_alm - std_times_no_alm, mean_times_no_alm + std_times_no_alm, color="orange", alpha=0.15)

        ax3.set_title("Mean OCP Resolution Time vs Training Iterations", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Training Iterations", fontsize=11)
        ax3.set_ylabel("Mean Resolution Time [s]", fontsize=11)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(fontsize=10)
    else:
        ax3.set_visible(False)

    plt.tight_layout()
    #plt.show()
    


def plot_violation_convergence(
    iters,
    mean_viol_alm, std_viol_alm, num_viol_alm,
    mean_viol_no_alm, std_viol_no_alm, num_viol_no_alm,
    to_episodes_alm=None,
    to_episodes_no_alm=None
):
    """
    Plots the convergence of the mean trajectory violation and number of violations over training iterations.

    Args:
        iters (list or np.ndarray): Training iterations.
        mean_viol_alm (list or np.ndarray): Mean violations for ALM policy.
        std_viol_alm (list or np.ndarray): Standard deviations of violations for ALM policy.
        num_viol_alm (list or np.ndarray): Number of violations for ALM policy.
        mean_viol_no_alm (list or np.ndarray): Mean violations for NO ALM policy.
        std_viol_no_alm (list or np.ndarray): Standard deviations of violations for NO ALM policy.
        num_viol_no_alm (list or np.ndarray): Number of violations for NO ALM policy.
        to_episodes_alm (list or np.ndarray, optional): Cumulative TO episodes solved for ALM policy. Defaults to None.
        to_episodes_no_alm (list or np.ndarray, optional): Cumulative TO episodes solved for NO ALM policy. Defaults to None.

    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # --- PLOT 1: Mean Violation vs Training Iterations ---
    ax1.plot(iters, mean_viol_alm, "o-", color="blue", label="ALM Policy", linewidth=2)
    #ax1.fill_between(iters, mean_viol_alm - std_viol_alm, mean_viol_alm + std_viol_alm, color="blue", alpha=0.15)
    
    ax1.plot(iters, mean_viol_no_alm, "s--", color="orange", label="NO ALM Policy", linewidth=2)
    #ax1.fill_between(iters, mean_viol_no_alm - std_viol_no_alm, mean_viol_no_alm + std_viol_no_alm, color="orange", alpha=0.15)

    ax1.set_title("Mean Violation vs Training Iterations", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Iterations", fontsize=11)
    ax1.set_ylabel("Mean Violation Depth", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # --- PLOT 2: Mean Violation vs TO Episodes Solved in Training ---
    if to_episodes_alm is not None and len(to_episodes_alm) > 0:
        ax2.plot(to_episodes_alm, mean_viol_alm, "o-", color="blue", label="ALM", linewidth=2)
        #ax2.fill_between(to_episodes_alm, mean_viol_alm - std_viol_alm, mean_viol_alm + std_viol_alm, color="blue", alpha=0.15)

        ax2.plot(to_episodes_no_alm, mean_viol_no_alm, "s--", color="orange", label="NO ALM", linewidth=2)
        #ax2.fill_between(to_episodes_no_alm, mean_viol_no_alm - std_viol_no_alm, mean_viol_no_alm + std_viol_no_alm, color="orange", alpha=0.15)

    ax2.set_title("Mean Violation vs TO Episodes Solved", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cumulative TO Episodes Solved in Training", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    # --- PLOT 3: Number of Violations vs Training Iterations ---
    ax3.plot(iters, num_viol_alm, "o-", color="blue", label="ALM", linewidth=2)
    ax3.plot(iters, num_viol_no_alm, "s--", color="orange", label="NO ALM", linewidth=2)

    ax3.set_title("Number of Constraint Violations vs Iterations", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Training Iterations", fontsize=11)
    ax3.set_ylabel("Total Number of Violating Trajectories", fontsize=11)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(fontsize=10)

    plt.tight_layout()