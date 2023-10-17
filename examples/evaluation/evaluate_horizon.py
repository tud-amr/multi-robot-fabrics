import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import copy
import sys
sys.path.insert(0, './')
from examples.example_pandas_Jointspace import run_panda_example, define_planners, define_rollout_planners
from examples.simulation_environments.create_simulation_manipulators import create_manipulators_simulation
from multi_robot_fabrics.utils.utils import UtilsKinematics
import examples.parameters_manipulators as parameters_manipulators


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
    # n_cubes = 4         #nr cubes per robot
    nr_robots = 2       #nr robots
    random_scene = True #random or predetermined scene
    cases = ["rollouts dynamic"]

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

    param = parameters_manipulators.manipulator_parameters(nr_robots=2, n_obst_per_link=1)
    simulation_class = create_manipulators_simulation(params=param)
    kinematics_class = UtilsKinematics()

    # --- first create n_runs randomizes obstacle environments: ---#
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
            RESOLVE_DEADLOCKS = True
        # -- make sure setting changes are saved! -- #
        param.define_settings(ROLLOUT_FABRICS=ROLLOUT_FABRICS,
                              ROLLOUTS_PLOTTING=ROLLOUTS_PLOTTING,
                              STATIC_OR_DYN_FABRICS=STATIC_OR_DYN_FABRICS,
                              RESOLVE_DEADLOCKS=RESOLVE_DEADLOCKS,
                              ESTIMATE_GOAL=ESTIMATE_GOAL,
                              N_HORIZON=N_HORIZON,
                              MPC_LAYER=False)

        # --- define planners and fk that is only changed for each case: ---#
        planners, planners_grasp, goal_structs = define_planners(params=param)
        fk_dict = kinematics_class.define_forward_kinematics(planners=planners, collision_links_nrs=param.collision_links_nrs, collision_links=param.collision_links)
        horizons = [1, 10, 20]
        results = []
        for h in horizons:
            param.set_horizon(h)
            if case == "rollouts static" or case == "rollouts dynamic" or case == "rollouts dynamic estimated":
                forwardplanner = define_rollout_planners(params=param, fk_dict=fk_dict, goal_structs=goal_structs)
            else:
                forwardplanner = None

            settings = [ROLLOUT_FABRICS, ROLLOUTS_PLOTTING, STATIC_OR_DYN_FABRICS, RESOLVE_DEADLOCKS,
                        ESTIMATE_GOAL, N_HORIZON]
            env = simulation_class.initialize_environment(render=render, random_scene=random_scene, obstacles=random_obstacles[0])
            res = run_panda_example(param, n_steps=n_steps, planners=planners, planners_grasp=planners_grasp,
                                    goal_structs=goal_structs,
                                    env=env, fk_dict=fk_dict, forwardplanner=forwardplanner)
            env.close()
            results.append(res)

        # --- store data --- #
        data = []
        for res in results:
            data.append(np.expand_dims(np.array(res['solver_times']),0))

        # --- open and display data --- #
        with open("results_horizon", "wb") as fp:  # Pickling
            pickle.dump(data, fp)

        file = open("results_horizon", 'rb')
        data_loaded = pickle.load(file)
        file.close()

        df = pd.DataFrame(data=np.concatenate(data_loaded,0).transpose(), columns=["K = " + str(x) for x in horizons])

        sns.boxplot(x="variable", y="value", data=pd.melt(df))
        plt.xlabel('')
        plt.ylabel('solver time [s]')
        plt.show()

        plt.savefig('figure_solver_time.png')

if __name__ == "__main__":
    n_steps = 100
    render = False
    n_runs = 1
    variables_plots = define_run_evaluations(n_steps=n_steps, render=render, n_runs=n_runs)


