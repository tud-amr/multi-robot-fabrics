import time
import sys
import yaml
sys.path.insert(0, './')
import copy
from examples.simulation_environments import create_simulation_manipulators
import casadi as ca
import numpy as np
from multi_robot_fabrics.others_planner.deadlock_prevention import deadlockprevention
from multi_robot_fabrics.utils.utils import UtilsKinematics
from multi_robot_fabrics.others_planner.state_machine import StateMachine
import examples.parameters_manipulators as parameters_manipulators
from multi_robot_fabrics.fabrics_planner.creating_planner import CreatingPlanner
from multi_robot_fabrics.utils.plot_rollouts import PlotRollouts

"""
Two panda example of a pick-and-place task. 
Rollouts are performed along the horizon where dynamic obstacles are forward simulated using their observed 
velocity in JOINT space. 
"""
class exampe_pandas_jointspace():
    def __init__(self):
        self.create_planners = CreatingPlanner()

    def define_obstacle_positions(self, ob, fk_fun_endeff, fk_fun_endeff_vel, params, fk_dict, dof: list, nr_robots: int=2):
        #define obstacle positions:
        x_robots = [[] for _ in range(nr_robots)]
        v_robots = [[] for _ in range(nr_robots)]
        a_robots = [[] for _ in range(nr_robots)]
        x_robots_ee = []
        v_robots_ee = []
        for i_robot in range(nr_robots):
            q_num = ob['robot_' + str(i_robot)]["joint_state"]["position"][0:dof[i_robot]]
            q_dot_num = ob['robot_' + str(i_robot)]["joint_state"]["velocity"][0:dof[i_robot]]
            # q_ddot_num = np.zeros((dof[i_robot], )) #currently no acceleration
            x_robots_ee.append(fk_fun_endeff[i_robot](q_num).full().transpose()[0])
            v_robots_ee.append(fk_fun_endeff_vel[i_robot](q_num).full().transpose()[0])

            for i_link in range(len(params.collision_links_nrs[i_robot])):
                x_robot = fk_dict["fk_fun"][i_robot][i_link](q_num)
                x_robots[i_robot].append(x_robot.full().transpose()[0])

                v_robot = fk_dict["jac_fun"][i_robot][i_link](q_num) @ q_dot_num
                if params.STATIC_OR_DYN_FABRICS == 0:
                    v_robots[i_robot].append(np.zeros((3,)))
                else:
                    v_robots[i_robot].append(v_robot.full().transpose()[0])

                # a_robot = fk_dict["jac_fun"][i_robot][i_link](q_num) @ q_ddot_num + \
                #              fk_dict["jac_dot_fun"][i_robot][i_link](q_num, q_dot_num) @ q_dot_num
                a_robots[i_robot].append(np.zeros((3,))) #currently no acceleration
        return x_robots, v_robots, a_robots, x_robots_ee, v_robots_ee

    def define_dynamic_obstacles(self, env, v_robots, params, nr_robots: int=2):
        x_dyns_obsts = [[] for _ in range(nr_robots)]
        v_dyns_obsts = [[] for _ in range(nr_robots)]
        a_dyns_obsts = [[] for _ in range(nr_robots)]
        r_dyns_obsts = [[] for _ in range(nr_robots)]
        x_dyns_obsts_per_robot = [[] for _ in range(nr_robots)]

        env.update_collision_links()
        x_collision_sphere_poses = env.collision_links_poses(position_only=True)
        for i_robot in range(nr_robots):
            i_other_robots = [i for i in range(nr_robots) if i != i_robot]
            x_dyns_obsts_per_robot[i_robot] = [x_sphere for key, x_sphere in x_collision_sphere_poses.items() if
                                               str(i_robot) in key[0]]
            for i_other_robot in i_other_robots:
                x_dyns_obsts[i_other_robot] = x_dyns_obsts[i_other_robot] + x_dyns_obsts_per_robot[i_robot]

                for i_sphere in range(len(v_robots[i_other_robot])):
                    v_dyns_obsts[i_robot] = v_dyns_obsts[i_robot] + [
                        v_robots[i_other_robot][i_sphere]] * params.n_obst_per_link
                    a_dyns_obsts[i_robot] = a_dyns_obsts[i_robot] + [
                        np.zeros((3,))] * params.n_obst_per_link  # [a_robots[i_other_robot][i_sphere]] * n_obst_per_link
                    r_dyns_obsts[i_robot] = r_dyns_obsts[i_robot] + [
                        params.r_robots[i_other_robot][i_sphere]] * params.n_obst_per_link
        return x_dyns_obsts, v_dyns_obsts, a_dyns_obsts, r_dyns_obsts, x_dyns_obsts_per_robot

    def run_panda_example(self, params, n_steps=5000, planners=[], planners_grasp=[], goal_structs=[], env=None, fk_dict=None, forwardplanner=None) -> dict:
        """
        Run the evaluations over time for the two robots.
        Outputs a dictionary of evaluation metrices.
        """
        if env is None:
            env = {}

        # --- Rename parameters --- #
        dof = params.dof

        # --- initialize lists/variables --- #
        n_steps_panda = np.NaN
        n_steps_panda2 = np.NaN
        success = [False, False]
        step_times = []
        solver_times = []
        min_clearance = 100
        constraints = [np.array([0, 0, 1, 0.0-params.mount_param["z_table"]])]*params.nr_robots
        nr_robots = len(params.collision_links_nrs)
        dof_index = [0]
        for i_robot in range(nr_robots):
            dof_index.append(dof_index[i_robot]+dof[0]+2)

        # --- rollout fabrics initialize ---#
        import socket
        # Define the server address and port
        server_address = ('127.0.0.1', 8080)

        # --- Velocity and acceleration limits --- #
        limits_action = ()
        limit_vel_panda = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61])
        for i_robot in range(nr_robots):
            limits_action = np.concatenate((limits_action, limit_vel_panda, np.array([2, 2])))

        # --- Initialize action and observation space --- #
        action = np.zeros(env.n())
        ob, *_ = env.step(action)

        # --- initialize symbolic forward kinematics ---#
        fk_fun_endeff = [[] for _ in range(nr_robots)]
        fk_fun_endeff_vel = [[] for _ in range(nr_robots)]
        q_pandas_sym = [[] for _ in range(nr_robots)]
        for i_robot in range(nr_robots):
            q_pandas_sym[i_robot] = planners[i_robot].variables.position_variable()
            endeff_panda = planners[i_robot].get_forward_kinematics("panda_hand")
            endeff_panda_vel = ca.jacobian(endeff_panda, q_pandas_sym[i_robot])
            fk_fun_endeff[i_robot] = ca.Function("fk_endeff", [q_pandas_sym[i_robot]], [endeff_panda])
            fk_fun_endeff_vel[i_robot] = ca.Function("fk_endeff_vel", [q_pandas_sym[i_robot]], [endeff_panda_vel])

        env.reconfigure_camera(2.5, -5., -42., (0.3, 1., -0.5))

        # forward fabrics
        if params.ROLLOUT_FABRICS == True:
            deadlock_prevention = deadlockprevention(dof, params.nr_robots, params.N_HORIZON)

        # state machine
        state_machines = []
        for i_robot in range(nr_robots):
            state_machine = StateMachine(start_goal=params.start_goals[i_robot],
                                         nr_robots=nr_robots,
                                         nr_blocks=params.n_cubes/nr_robots,
                                         fk_fun_ee=fk_fun_endeff[i_robot],
                                         robot_types=params.robot_types)
            state_machines.append(state_machine)

        # Initialize iteration variables:
        q_pandas = [[] for _ in range(nr_robots)]
        qdot_pandas = [[] for _ in range(nr_robots)]
        q_pandas_gripper = [[] for _ in range(nr_robots)]
        ob_pandas = [[] for _ in range(nr_robots)]
        state_machine_pandas = [[] for _ in range(nr_robots)]
        goal_pandas = [[] for _ in range(nr_robots)]
        goal_weights = [[] for _ in range(nr_robots)]
        nr_blocks_pandas_success = [0 for _ in range(nr_robots)]
        goal_pandas_block = [[] for _ in range(nr_robots)]
        time_deadlock_out = 1000

        # --- Avoid longer computation time due to previously running processes --- #
        time.sleep(1)
        self.plot_rollouts =PlotRollouts(params=params)

        # --- Loop over all steps over time -- #
        for w in range(n_steps):

            t_start_loop = time.perf_counter()

            # --- define states --- #
            for i_robot in range(nr_robots):
                ob_pandas[i_robot] = ob['robot_'+str(i_robot)]
                q_pandas[i_robot] = ob_pandas[i_robot]["joint_state"]["position"][0:dof[0]]
                q_pandas_gripper[i_robot] = ob_pandas[i_robot]["joint_state"]["position"][dof[0]:dof[0]+2]
                qdot_pandas[i_robot] = np.clip(ob_pandas[i_robot]["joint_state"]["velocity"][0:dof[0]], -limit_vel_panda, limit_vel_panda)

                # --- check if blocks are already picked ---#
                nr_blocks_pandas_success[i_robot] = state_machines[i_robot].get_nr_blocks_picked()

                # --- update block positions --- #
                first_index = list(ob_pandas[0]['FullSensor']['obstacles'].keys())[0]
                if nr_blocks_pandas_success[i_robot]<params.n_cubes/nr_robots:
                    goal_pandas_block[i_robot] = copy.deepcopy(ob_pandas[0]['FullSensor']['obstacles'][first_index+nr_blocks_pandas_success[i_robot]+int(i_robot*(params.n_cubes/nr_robots))]['position'])
                    goal_pandas_block[i_robot][2] += 0.1

            # --- update state machine --- #
            for i_robot in range(nr_robots):
                state_machine_pandas[i_robot] = state_machines[i_robot].get_state_machine_panda(q_robot=q_pandas[i_robot], q_robot_gripper=q_pandas_gripper[i_robot], goal_block=goal_pandas_block[i_robot], robot_type="panda")

            #todo: adapt to 3 robot case!!
            if state_machine_pandas[0] == 10 and success[0] == False:
                n_steps_panda = w
                success[0] = True
            if state_machine_pandas[1] == 10 and success[1] == False:
                n_steps_panda2 = w
                success[1] = True
            if all([state_machine_pandas[i_robot] == 10 for i_robot in range(nr_robots)]):
                break

            # --- Get new goals and goal weights depending on the state machine --- #
            for i_robot in range(nr_robots):
                goal_pandas[i_robot] = state_machines[i_robot].get_goal_robot()
                goal_weights[i_robot] = state_machines[i_robot].get_weight_goal0()

            #define obstacle positions:
            x_robots, v_robots, a_robots, x_robots_ee, v_robots_ee = self.define_obstacle_positions(ob, fk_fun_endeff, fk_fun_endeff_vel, params, fk_dict, dof, nr_robots)

            # --- Estimation with constant velocity -- #
            if params.ESTIMATE_GOAL == True:
                goal_panda2_estimated = x_robots_ee[1] + 20 * 0.01 * v_robots_ee[1]
                goal_pandas[1] = goal_panda2_estimated

            t_rollouts = 0

            if params.ROLLOUT_FABRICS:
                t_start_rollouts = time.perf_counter()
                inputs_action = {"q_robots": q_pandas, "q_dot_robots": qdot_pandas,
                                 "x_obsts": [[]*nr_robots],
                                 # "radius_obsts": [[0.1], [0.1]],
                                 # "x_robots": x_robots,
                                 # "v_robots": v_robots,
                                 # "a_robots": a_robots,
                                 "x_goals0": goal_pandas,
                                 "x_goals1": [goal_structs[i_robot]._config.subgoal1.desired_position for i_robot in range(nr_robots)],
                                 "x_goals2": [goal_structs[i_robot]._config.subgoal2.desired_position for i_robot in range(nr_robots)],
                                 "weight_goals0": goal_weights,
                                 "weight_goals1": [goal_structs[i_robot]._config.subgoal1.weight for i_robot in range(nr_robots)],
                                 "weight_goals2": [goal_structs[i_robot]._config.subgoal2.weight for i_robot in range(nr_robots)],
                                 # "r_robots": params.r_robots,
                                 # "angle_goals1": params.rotation_matrix_pandas,
                                 "constraints": constraints}

                time_avg_vel_start = time.perf_counter()
                vel_avg = forwardplanner.get_velocity_rollouts(inputs_action=inputs_action)
                vel_avg_tot = sum(vel_avg)/nr_robots
                time_avg_vel = time.perf_counter() - time_avg_vel_start
                # print("time_avg_vel:", time_avg_vel)

                goal_robots, goal_weights, time_deadlock_out = deadlock_prevention.deadlock_checking(x_robots=x_robots_ee,
                                                                                                     goal_robots=goal_pandas,
                                                                                                     goal_weights=goal_weights,
                                                                                                     time_step=w,
                                                                                                     time_deadlock_out=time_deadlock_out,
                                                                                                     avg_sum=vel_avg_tot,
                                                                                                     state_machine_robots=state_machine_pandas)
                t_rollouts = (time.perf_counter() - t_start_rollouts)

            x_dyns_obsts, v_dyns_obsts, a_dyns_obsts, r_dyns_obsts, x_dyns_obsts_per_robot = self.define_dynamic_obstacles(env, v_robots, params, nr_robots)

            t_start_actions = time.perf_counter()

            # --- compute action ---#
            for i_robot in range(nr_robots):
                if state_machine_pandas[i_robot] == 3 or state_machine_pandas[i_robot] == 5:
                    action[dof_index[i_robot]:dof_index[i_robot]+dof[i_robot]] = np.zeros(dof[0])
                else:
                    arguments_robot = dict(q=q_pandas[i_robot],
                                           qdot=qdot_pandas[i_robot],
                                           x_goal_0=np.array(goal_pandas[i_robot]),
                                           weight_goal_0=goal_weights[i_robot],
                                           angle_goal_1=params.rotation_matrix_pandas[i_robot],
                                           x_goal_1=np.array([0.107, 0.0, 0.0]),
                                           weight_goal_1=20.0,
                                           x_goal_2=np.array([np.pi / 4]),
                                           weight_goal_2=1.0,
                                           x_obsts=x_dyns_obsts[i_robot],
                                           radius_obsts=r_dyns_obsts[i_robot],
                                           constraint_0=constraints[i_robot],
                                           radius_body_panda_links=params.radius_body_panda_links,
                                           radius_body_panda_hand=np.array([params.radius_sphere]),
                                           x_obsts_dynamic=x_dyns_obsts[i_robot],
                                           xdot_obsts_dynamic=v_dyns_obsts[i_robot],
                                           xddot_obsts_dynamic=a_dyns_obsts[i_robot],
                                           radius_obsts_dynamic=r_dyns_obsts[i_robot],
                                           )
                    if state_machine_pandas[i_robot] == 2:  # only goal reaching, no obstacle avoidance
                        action[dof_index[i_robot]:dof_index[i_robot]+dof[i_robot]] = planners_grasp[i_robot].compute_action(**arguments_robot)
                    else:
                        time_action_start = time.perf_counter()
                        action[dof_index[i_robot]:dof_index[i_robot]+dof[i_robot]] = planners[i_robot].compute_action(**arguments_robot)
                        time_action_diff = time.perf_counter() - time_action_start
                # gripper action
                action[dof_index[i_robot]+dof[i_robot]:dof_index[i_robot+1]] = state_machines[i_robot].get_gripper_action_panda(q_pandas_gripper[i_robot])

            t_actions = (time.perf_counter() - t_start_actions)/2

            # --- Apply action --- #
            action = np.clip(action, -limits_action, limits_action)
            ob, *_ = env.step(action)

            t_end_loop = time.perf_counter()
            solver_times = np.append(solver_times, t_actions+t_rollouts)
            step_times = np.append(step_times, t_end_loop - t_start_loop)

            # Collision sphere violation: check if the robot is within a collision sphere of the other robot.
            for k, x_panda_1 in enumerate(x_dyns_obsts_per_robot[0]):
                for j, x_panda_2 in enumerate(x_dyns_obsts_per_robot[1]):
                    dist_x = np.linalg.norm(x_panda_1 - x_panda_2, 2)
                    dist_x_r = dist_x - r_dyns_obsts[0][k] - r_dyns_obsts[1][k]  #todo!!!: make sure radius matches
                    if dist_x_r < min_clearance:
                        min_clearance = dist_x_r
                    # clearances.append(dist_x_r)
                    # clearances_episode.append(dist_x_r)
                    if dist_x_r < 0:
                        print("clearance violation:", dist_x_r)

            # --- for plotting the rollouts and checking if the estimated states match the real states --- #
            if params.ROLLOUTS_PLOTTING == True and params.ROLLOUT_FABRICS == True:
                self.plot_rollouts.update_variable_lists(q_pandas, qdot_pandas, action, w, forwardplanner, inputs_action)

        if params.ROLLOUTS_PLOTTING == True:
            self.plot_rollouts.plot_results()

        solver_time_mean = np.mean(solver_times)
        solver_time_std = np.std(solver_times)
        step_time_mean = np.mean(step_times)
        step_time_std = np.std(step_times)
        total_time = max([n_steps_panda, n_steps_panda2])*0.01
        success_rate = state_machine.get_success_rate()

        #  env.stop_video_recording()
        return {'success_rate':success_rate, 'n_steps_panda':n_steps_panda, 'n_steps_robot2': n_steps_panda2,
                'step_time_mean': step_time_mean, "step_time_std": step_time_std,
                'total_time': total_time, 'dt': params.dt,
                "solver_time_mean": solver_time_mean, "solver_time_std": solver_time_std,
                "min clearance": min_clearance,
                "solver_times": solver_times
        }

    def define_run_panda_example(self, n_steps=100, render=True):
        with open("examples/configs/panda_config.yaml", "r") as setup_stream:
            setup = yaml.safe_load(setup_stream)
        random_scene = False
        nr_robots = setup['n_robots']
        param = parameters_manipulators.manipulator_parameters(nr_robots=nr_robots, n_obst_per_link=setup['n_obst_per_link'])
        simulation_class = create_simulation_manipulators.create_manipulators_simulation(param)
        utils_class = UtilsKinematics()
        random_obstacles = simulation_class.create_scene(random_scene=random_scene, n_cubes=param.n_cubes)
        env = simulation_class.initialize_environment(render=render, random_scene=random_scene, obstacles=random_obstacles)
        planners, planners_grasp, goal_structs = self.create_planners.define_planners(params=param)
        fk_dict = utils_class.define_forward_kinematics(planners=planners, collision_links=param.collision_links, collision_links_nrs=param.collision_links_nrs)
        settings = param.define_settings(ROLLOUT_FABRICS = setup['ROLLOUT_FABRICS'], ROLLOUTS_PLOTTING = setup['ROLLOUTS_PLOTTING'],
                                         STATIC_OR_DYN_FABRICS=setup['STATIC_OR_DYN_FABRICS'], RESOLVE_DEADLOCKS=setup['RESOLVE_DEADLOCKS'],
                                         ESTIMATE_GOAL=setup['ESTIMATE_GOAL'], N_HORIZON=setup['N_HORIZON'], n_obst_per_link=setup['n_obst_per_link'])
        if param.ROLLOUT_FABRICS == True:
            forwardplanner = self.create_planners.define_rollout_planners(params=param, fk_dict=fk_dict, goal_structs=goal_structs)
        else:
            forwardplanner = None
        res = self.run_panda_example(params=param, n_steps=n_steps, planners=planners, planners_grasp=planners_grasp, goal_structs=goal_structs, env=env, fk_dict=fk_dict, forwardplanner=forwardplanner)
        env.close()
        return res


if __name__ == "__main__":
    n_steps = 7000
    render = True
    pandasjointspace = exampe_pandas_jointspace()
    res = pandasjointspace.define_run_panda_example(n_steps=n_steps, render=render)