from examples.example_pandas_Jointspace import run_panda_example, define_planners, define_rollout_planners
import examples.parameters_manipulators
from examples.simulation_environments.create_simulation_manipulators import create_manipulators_simulation
from multi_robot_fabrics.utils.utils import UtilsKinematics
from texttable import Texttable
import latextable
import numpy as np
import pickle
import copy

"""
RUN EVALUATIONS WITH DYNAMIC FABRICS, ROLLOUT FABRICS AND ESTIMATED ROLLOUT FABRICS
NOTE: since this script takes very long for 50 runs with 7000 steps, change it back to:
    n_steps = 7000
    n_runs = 1
for a reasonable computation time ;)

When running this file, n_runs of random scenarios are evaluated for a scenario of two manipulators performing
a pick-and-place task. 
This is done with:
1) Dynamic fabrics
2) Rollout dynamic fabrics
3) Rollout dynamic fabrics with goal estimation (limited communication)

The solver times are stored in a file (more can be stored if desired)
and a latex table is printed with all mean and standard deviations of important evaluation metrices. 
"""

def get_std(list_of_std: list) -> float:
    """
    input: list of standard deviations
    output: Average standard deviation
    """
    variance_sum = 0
    for std in list_of_std:
        variance_sum = variance_sum + std**2
    std_avg = np.sqrt(variance_sum/len(list_of_std))
    return std_avg

def define_run_evaluations(n_steps=100, render=False, n_runs=1):
    # --- initialize variables ---#
    nr_robots = 2       #nr robots
    random_scene = True #random or predetermined scene
    cases = ["dynamic", "rollouts dynamic", "rollouts dynamic estimated", ]

    # --- initialize lists --- #
    bibl_dyn_num = {"dynamic": 0, "rollouts dynamic": 0,  "rollouts dynamic estimated":0}
    bibl_dyn_list= {"dynamic": [], "rollouts dynamic": [], "rollouts dynamic estimated":[]}
    n_success = copy.deepcopy(bibl_dyn_num)
    nr_collision_episodes_all = copy.deepcopy(bibl_dyn_num)
    time2success_all = copy.deepcopy(bibl_dyn_list)
    min_clearance_all = copy.deepcopy(bibl_dyn_list)
    step_time_all = copy.deepcopy(bibl_dyn_list)
    solver_time_all = copy.deepcopy(bibl_dyn_list)
    step_time_std = copy.deepcopy(bibl_dyn_list)
    solver_time_std = copy.deepcopy(bibl_dyn_list)
    success_total = copy.deepcopy(bibl_dyn_list)

    param = examples.parameters_manipulators.manipulator_parameters(nr_robots=2)
    simulation_class = create_manipulators_simulation(params=param)
    kinematics_class = UtilsKinematics()

    # --- first create n_runs randomizes obstacle environments --- #
    random_obstacles = []
    for _ in range(n_runs):
        obstacles = simulation_class.create_scene(random_scene, n_cubes = param.n_cubes)
        random_obstacles.append(obstacles)

    # --- predefine the planners --- #
    for case in cases:
        # -- define variables per case -- #
        [ROLLOUT_FABRICS, ROLLOUTS_PLOTTING, STATIC_OR_DYN_FABRICS, RESOLVE_DEADLOCKS, ESTIMATE_GOAL, N_HORIZON, MPC_LAYER] = param.get_settings()
        if case == "dynamic" or case == "rollouts dynamic" or case == "rollouts dynamic estimated":
            STATIC_OR_DYN_FABRICS = 1
        if case == "rollouts static" or case == "rollouts dynamic" or case == "rollouts dynamic estimated":
            ROLLOUT_FABRICS = True
        if case == "rollouts dynamic estimated":
            ESTIMATE_GOAL = True
            RESOLVE_DEADLOCKS = False
        # -- make sure setting changes are saved! -- #
        param.define_settings(ROLLOUT_FABRICS=ROLLOUT_FABRICS,
                          ROLLOUTS_PLOTTING=ROLLOUTS_PLOTTING,
                          STATIC_OR_DYN_FABRICS=STATIC_OR_DYN_FABRICS,
                          RESOLVE_DEADLOCKS=RESOLVE_DEADLOCKS,
                          ESTIMATE_GOAL=ESTIMATE_GOAL,
                          N_HORIZON=N_HORIZON,
                          MPC_LAYER=MPC_LAYER)

        # --- define planners and fk that is only changed for each case: ---#
        planners, planners_grasp, goal_structs = define_planners(params=param)
        fk_dict = kinematics_class.define_forward_kinematics(planners, collision_links_nrs=param.collision_links_nrs, collision_links=param.collision_links)
        if case == "rollouts static" or case == "rollouts dynamic" or case == "rollouts dynamic estimated":
            forwardplanner = define_rollout_planners(param, fk_dict=fk_dict, goal_structs=goal_structs, n_steps=100)
        else:
            forwardplanner = None

        # --- runs --- #
        results = []
        for z in range(n_runs):
            env = simulation_class.initialize_environment(render=render, random_scene=random_scene, obstacles=random_obstacles[z])
            res = run_panda_example(param, n_steps=n_steps, planners=planners, planners_grasp=planners_grasp,
                                goal_structs=goal_structs,
                                env=env, fk_dict=fk_dict, forwardplanner=forwardplanner)
            env.close()
            results.append(res)

            n_success[case] = n_success[case] + res["success_rate"]
            success_total[case].append(res["success_rate"])

            # ---- save results --- #
            time2success_all[case].append(np.nanmax([res['n_steps_panda'], res['n_steps_robot2']])*res['dt'])
            if res["success_rate"]  == 1:
                min_clearance_all[case].append([res["min clearance"]])
                if res["min clearance"]<0:
                    nr_collision_episodes_all[case] = nr_collision_episodes_all[case] + 1
            solver_time_all[case].append(res["solver_time_mean"])
            step_time_all[case].append(res['step_time_mean'])
            solver_time_std[case].append(res["solver_time_std"])
            step_time_std[case].append(res['step_time_std'])

        # --- store data --- #
        data = []
        for res in results:
            data.append(np.expand_dims(np.array(res["solver_times"]), 0))

        with open("results_dynamic_scenarios", "wb") as fp:
            pickle.dump(data, fp)

    # --- create and plot table --- #
    rows = []
    rows.append([' ','Time-to-Success', "# Collision Episodes","Min Clearance", 'Solver-Time', "Step-Time", 'Success-Rate'])
    for case in cases:
        if len(min_clearance_all[case])>0:
            collision_episodes_rate = nr_collision_episodes_all[case] / len(min_clearance_all[case])
        else:
            collision_episodes_rate = 0

        rows.append([case,
                     str(np.round(np.nanmean(time2success_all[case]), decimals=4)) + "+-" + str(np.round(np.nanstd(time2success_all[case]), decimals=4)),
                     str(np.round(collision_episodes_rate, decimals=8)),
                     str(np.round(np.mean(min_clearance_all[case]), decimals=4)) + "+-" + str(np.round(np.nanstd(min_clearance_all[case]), decimals=4)),
                     str(np.round(np.nanmean(solver_time_all[case]), decimals=4)) + "+-" + str(np.round(get_std(solver_time_std[case]), decimals=4)),
                     str(np.round(np.nanmean(step_time_all[case]), decimals=4)) + "+-" + str(np.round(get_std(step_time_std[case]), decimals=4)),
                     str(np.round(n_success[case] / n_runs, decimals=4)) + "+-" + str(np.round(np.nanstd(success_total[case]), decimals=4))])

    table = Texttable()
    table.set_cols_align(["c"] * 7)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)

    print('\nTexttable Latex:')
    print(latextable.draw_latex(table, caption="Statistics for 200 runs of our proposed method compared to"))

if __name__ == "__main__":
    n_steps = 7000
    n_runs = 50
    render = False
    variables_plots = define_run_evaluations(n_steps=n_steps, render=render, n_runs=n_runs)

