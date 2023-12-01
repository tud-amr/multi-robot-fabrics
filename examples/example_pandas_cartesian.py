import time
import copy
import yaml
from examples.simulation_environments import create_simulation_manipulators
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from mpscenes.goals.goal_composition import GoalComposition
import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from multi_robot_fabrics.others_planner.deadlock_prevention import deadlockprevention
from multi_robot_fabrics.others_planner.state_machine import StateMachine
import examples.parameters_manipulators
from multi_robot_fabrics.fabrics_planner.forward_planner_Cartesian import FabricsRollouts
from multi_robot_fabrics.utils.utils import UtilsKinematics
from multi_robot_fabrics.utils.generate_figures import generate_plots
from multi_robot_fabrics.utils.utils_apply_fk import compute_x_obsts_dyn_0, compute_endeffector

"""
!!! IMPORTANT !!!
Compared to example_pandas_Jointspace.py, the forward rollouts of positions and velocities of the obstacles/other agents 
are NOT done in joint-space/configuration space, but in CARTESIAN space.
This simplifies the computation of the obstacle positions and velocities at the next time-instance.
"""

def create_dummy_goal_panda() -> GoalComposition:
    """
    Create a dummy goal where the numbferical values of the dictionary can later be changed.
    """
    goal_orientation = [-0.366, 0.0, 0.0, 0.3305]
    whole_position = [0.1, 0.6, 0.8]
    goal_dict = {
        "subgoal0": {
            "weight": 2.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "world",
            "child_link": "panda_hand",
            "desired_position": whole_position,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "weight": 20.0,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "panda_link7",
            "child_link": "panda_hand",
            "desired_position": [0.107, 0.0, 0.0],
            "angle": goal_orientation,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal2": {
            "weight": 1.0,
            "is_primary_goal": False,
            "indices": [6],
            "desired_position": [np.pi/4],
            "epsilon": 0.05,
            "type": "staticJointSpaceSubGoal",
        },
    }
    return GoalComposition(name="goal", content_dict=goal_dict)

def set_planner_panda(degrees_of_freedom: int = 7, nr_obst = 0, nr_obst_dyn=1, collision_links_nr= [5], urdf_links = {}, mount_transform = [], i_robot=0):
    """
    Initializes the fabric planner for the panda robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """
    goal = create_dummy_goal_panda()
    with open(urdf_links["URDF_file_panda"], 'r') as file:
        urdf = file.read()
    fk = GenericURDFFk(urdf, 'panda_link0', 'panda_leftfinger')
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        fk,
        geometry_plane_constraint="10*(1/(1+1*ca.exp(-10*x))-1) * (xdot**2)",
        collision_geometry = "-0.5 / (x ** 4) * (xdot ** 2)",
        collision_finsler = "0.01/(x**4) * xdot**2",
    )
    collision_links = []
    for col_link_i in collision_links_nr:
        if col_link_i<9:
            collision_links.append('panda_link' + str(col_link_i))
        else:
            collision_links.append('panda_hand')
    panda_limits = [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ]

    # --- mount position/orientation: rotation angle and corresponding transformation matrix ---#
    planner._forward_kinematics.set_mount_transformation(mount_transformation=mount_transform[i_robot])

    # self_collision_pairs = {"panda_link6": ["panda_link3"]} #syntax for self-collision pairs.

    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        # self_collision_pairs=self_collision_pairs,
        collision_links=collision_links,
        goal=goal,
        number_obstacles=nr_obst,
        number_dynamic_obstacles=nr_obst_dyn,
        dynamic_obstacle_dimension=3,
        number_plane_constraints=1,
        limits=panda_limits,
    )
    planner.concretize(mode='vel', time_step=0.01)  # be careful that the inputs are here VELOCITIES!!!
    return planner, goal

def define_planners(params):
    """
    Define the planners symbolically for each robot
    A separate planner that handles the grasping of a block (from pregrasp to grasp position)
    """
    if params.STATIC_OR_DYN_FABRICS == 0:
        nr_obst_planners = params.nr_obsts_dyn_all
        nr_obst_dyn_planners = [0]*params.nr_robots
    else:
        nr_obst_planners = [0]*params.nr_robots
        nr_obst_dyn_planners = params.nr_obsts_dyn_all

    # --- Planners for the robots  --- #
    planners = []
    goal_structs = []
    planners_grasp = []
    for i_robot in range(params.nr_robots):
        planner_panda_i, goal_struct_panda_i = set_planner_panda(degrees_of_freedom=params.dof[i_robot],
                                                                nr_obst=nr_obst_planners[i_robot],
                                                                nr_obst_dyn=nr_obst_dyn_planners[i_robot],
                                                                collision_links_nr=params.collision_links_nrs[i_robot],
                                                                urdf_links = params.urdf_links,
                                                                mount_transform = params.mount_transform,
                                                                i_robot=i_robot)
        planner_panda_grasp_i, _= set_planner_panda(degrees_of_freedom=params.dof[i_robot],
                                                    nr_obst=0,
                                                    nr_obst_dyn=0,
                                                    collision_links_nr=[],
                                                    urdf_links=params.urdf_links,
                                                    mount_transform=params.mount_transform,
                                                    i_robot=i_robot)
        planners.append(planner_panda_i)
        goal_structs.append(goal_struct_panda_i)
        planners_grasp.append(planner_panda_grasp_i)
    return planners, planners_grasp, goal_structs

def define_rollout_planners(params, fk_dict=None, goal_structs=None, n_steps=100, planners=[], nr_robots=2):
    """
    Define the planners that are used for the rollouts (less obstacles per link)
    """
    forwardplanners = []

    # for computational overhead, it is possible to define different planners for the rollouts,
    # with only 1 collision link per sphere. Can be easily changed:
    # planners_fabrics_rollout = []
    # for i_robot in range(params.nr_robots):
    #     planner_panda_fk_i, goal = set_planner_panda(degrees_of_freedom=params.dof[i_robot],
    #                                               nr_obst=params.nr_obsts[i_robot],
    #                                               nr_obst_dyn=params.nr_obsts_dyn[i_robot],
    #                                               collision_links_nr=params.collision_links_nrs[i_robot],
    #                                               urdf_links=params.urdf_links,
    #                                               mount_transform=params.mount_transform,
    #                                               i_robot=i_robot)
    #     planners_fabrics_rollout.append(planner_panda_fk_i)

    v_obsts_dyn = [np.zeros((3,))]* params.nr_obsts_dyn_all[0]

    for i_robot in range(nr_robots):
        forwardplanner_i = FabricsRollouts(N=params.N_HORIZON, dt=params.dt, nx=params.dof[i_robot]*2, nu=params.dof[i_robot],
                                        dof=params.dof[i_robot], nr_obsts=params.nr_obsts[i_robot],
                                        bool_ring=False, nr_obsts_dyn=params.nr_obsts_dyn_all[i_robot],
                                        v_obsts_dyn=v_obsts_dyn, fabrics_mode=params.fabrics_mode,
                                        collision_links_nrs=params.collision_links_nrs[i_robot],
                                        nr_constraints=params.nr_constraints[i_robot], radius_sphere=params.radius_sphere,
                                        constraints=params.constraints[i_robot],
                                        nr_goals=len(goal_structs[i_robot]._config))
        forwardplanner_i.symbolic_forward_fabrics(planner=planners[i_robot], goal_struct=goal_structs[i_robot])
        forwardplanners.append(forwardplanner_i)
    return forwardplanners

def run_panda_example(params, n_steps=5000, planners=[], planners_grasp=[], goal_structs=[], env=None, fk_dict=None, forwardplanners=None, fk_dict_spheres=None, utils_class=None) -> dict:
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
    # constraints = [np.array([0, 0, 1, 0.0-params.mount_param["z_table"]])]*params.nr_robots
    nr_robots = len(params.collision_links_nrs)
    dof_index = [0]
    for i_robot in range(nr_robots):
        dof_index.append(dof_index[i_robot]+dof[0]+2)

    # --- Velocity and acceleration limits --- #
    limits_action = ()
    limit_vel_panda = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61])
    for i_robot in range(nr_robots):
        limits_action = np.concatenate((limits_action, limit_vel_panda, np.array([2, 2])))

    # --- Initialize action and observation space --- #
    action = np.zeros(env.n())
    ob, *_ = env.step(action)

    # --- initialize symbolic forward kinematics ---#
    fk_endeff = utils_class.define_symbolic_endeffector(planners)

    # --- Camera angle ---- #
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
                                     fk_fun_ee=fk_endeff[i_robot]["fk_fun_ee"],
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

    # Initialize lists for plotting
    state_j = [ [] for _ in range(dof[0]) ]
    vel_j = [ [] for _ in range(dof[0]) ]
    acc_j = [ [] for _ in range(dof[0]) ]
    q_robots_N = {}
    q_dot_robots_N = {}
    q_ddot_robots_N = {}
    q_num_N = {}
    q_dot_num_N = {}
    q_ddot_num_N = {}
    x_obsts_dyn_N_num = {}
    x_obsts_dyn_N = {}
    weight_goals = {}
    x_goals = {}
    for i_robot in range(nr_robots):
        weight_goals["robot_"+str(i_robot)] = {}
        x_goals["robot_"+str(i_robot)] = {}
    q_N_time_j = [ [] for _ in range(dof[0])]
    q_dot_N_time_j = [ [] for _ in range(dof[0])]
    q_ddot_N_time_j = [[] for _ in range(dof[0])]
    q_MPC_N_time_j = [ [] for _ in range(dof[0])]
    q_dot_MPC_N_time_j = [ [] for _ in range(dof[0])]
    q_ddot_MPC_N_time_j = [[] for _ in range(dof[0])]
    vel_avg = [[] for _ in range(nr_robots)]
    pos_xyz = []
    time_deadlock_out = 1000
    exitflag_list = [[] for _ in range(nr_robots)]
    if params.ROLLOUT_FABRICS == True:
        for i_robot in range(nr_robots):
            forwardplanners[i_robot].preset_radii_obsts_dyn(radii_obst_dyn=params.r_dyns_obsts[i_robot])

    # --- Avoid longer computation time due to previously running processes --- #
    time.sleep(1)


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
            key_i = "robot_"+str(i_robot)
            goal_pandas[i_robot] = state_machines[i_robot].get_goal_robot()
            goal_weights[i_robot] = state_machines[i_robot].get_weight_goal0()

            for i_subgoal in range(len(goal_structs[i_robot]._config)):
                if i_subgoal == 0:
                    weight_goals[key_i]["subgoal" + str(i_subgoal)] = goal_weights[i_robot]
                    x_goals[key_i]["subgoal" + str(i_subgoal)]= goal_pandas[i_robot]
                else:
                    weight_goals[key_i]["subgoal" + str(i_subgoal)] = goal_structs[i_robot]._config["subgoal" + str(i_subgoal)]["weight"]
                    x_goals[key_i]["subgoal" + str(i_subgoal)] = goal_structs[i_robot]._config["subgoal" + str(i_subgoal)]["desired_position"]

        #define obstacle positions and velocities:
        env.update_collision_links()
        x_collision_sphere_poses = env.collision_links_poses(position_only=True)
        x_dyns_obsts, v_dyns_obsts, x_dyns_obsts_per_robot = compute_x_obsts_dyn_0(q_robots=q_pandas, qdot_robots=qdot_pandas,
                                                          x_collision_sphere_poses=x_collision_sphere_poses,
                                                          nr_robots=nr_robots,
                                                          fk_dict_spheres=fk_dict_spheres,
                                                          nr_dyn_obsts = params.nr_obsts_dyn_all)

        #define end-effector positions and velocities:
        x_robots_ee, v_robots_ee = compute_endeffector(q_pandas, qdot_pandas, fk_endeff, nr_robots=params.nr_robots)
        pos_xyz.append(x_robots_ee[0])
        # --- Estimation with constant velocity -- #
        if params.ESTIMATE_GOAL == True:
            goal_panda2_estimated = x_robots_ee[1] + 20 * 0.01 * v_robots_ee[1]
            x_goals["robot_1"]["subgoal0"] = goal_panda2_estimated

        t_rollouts = 0

        if params.ROLLOUT_FABRICS:
            t_start_rollouts = time.perf_counter()
            arguments = [[] for _ in range(nr_robots)]
            for i_robot in range(nr_robots):
                # forwardplanners[i_robot].reset_v_obsts_dyn(v_dyns_obsts)
                arguments[i_robot] = forwardplanners[i_robot].define_arguments_numerical(q_robot=q_pandas[i_robot],
                                                                    q_dot_robot=qdot_pandas[i_robot],
                                                                    constraints=params.constraints[i_robot],
                                                                    weight_goals=weight_goals["robot_" + str(i_robot)],
                                                                    x_goals=x_goals["robot_" + str(i_robot)],
                                                                    x_obsts=[],
                                                                    x_obsts_dyn=x_dyns_obsts[i_robot],
                                                                    v_obsts_dyn=v_dyns_obsts[i_robot])
                # --- The numerical rollouts of state and action along the horizon --- #
                if params.ROLLOUTS_PLOTTING == True:
                    # ---- forward fabrics ----- #
                    key_i = "robot_"+str(i_robot)
                    ## ---- numerically evaluation using symbolic function --- #
                    [q_robots_N[key_i], q_dot_robots_N[key_i], q_ddot_robots_N[key_i]] = forwardplanners[i_robot].rollouts_numerical(arguments[i_robot])
                    if params.nr_obsts_dyn[i_robot] > 0:
                        x_obsts_dyn_N[key_i] = forwardplanners[i_robot].x_obsts_dyn_numerical(pos_obsts_dyn=x_dyns_obsts[i_robot])
                    else:
                        x_obsts_dyn_N[key_i] = [[] for _ in range(params.N_HORIZON)]

                    #   ----- numerically evaluations (done realtime, not constructed symbolically) ----- #
                    [q_num_N[key_i], q_dot_num_N[key_i], q_ddot_num_N[key_i]] = forwardplanners[i_robot].forward_fabrics(planner=planners[i_robot],
                                                                                                    pos_k=q_pandas[i_robot],
                                                                                                    vel_k=qdot_pandas[i_robot],
                                                                                                    ob_robot=ob_pandas[i_robot],
                                                                                                    goal=goal_structs[i_robot],
                                                                                                    x_obsts_dyn_0=x_dyns_obsts[i_robot],
                                                                                                    x_goals_struct=x_goals["robot_"+str(i_robot)],
                                                                                                    weight_goals_struct=weight_goals["robot_"+str(i_robot)])
                    if params.nr_obsts_dyn[i_robot] > 0:
                        x_obsts_dyn_0_N, _ = forwardplanners[i_robot].get_x_obsts_dyn_N(x_obsts_dyn=x_dyns_obsts[i_robot])
                        x_obsts_dyn_N_num[key_i] = x_obsts_dyn_0_N[1:params.N_HORIZON+1]

            t_rollouts = (time.perf_counter() - t_start_rollouts)

            # --- For checking if a deadlock is present along the horizon ---#
            if params.RESOLVE_DEADLOCKS == True:
                time_avg_vel_start = time.perf_counter()
                for i_robot in range(nr_robots):
                    vel_avg[i_robot] = forwardplanners[i_robot].get_velocity_rollouts(arguments[i_robot]).full()[0]
                vel_avg_tot = sum(vel_avg)/nr_robots
                time_avg_vel = time.perf_counter() - time_avg_vel_start

                goal_deadl, weight_deadl, time_deadlock_out = deadlock_prevention.deadlock_checking(
                        x_robots=x_robots_ee,
                        goal_robots=[x_goals["robot_"+str(i_robot)]["subgoal0"] for i_robot in range(nr_robots)],
                        goal_weights=[weight_goals["robot_"+str(i_robot)]["subgoal0"] for i_robot in range(nr_robots)],
                        time_step=w,
                        time_deadlock_out=time_deadlock_out,
                        avg_sum=vel_avg_tot,
                        state_machine_robots=state_machine_pandas)
                for i_robot in range(nr_robots):
                    x_goals["robot_"+str(i_robot)]["subgoal0"] = goal_deadl[i_robot]
                    weight_goals["robot_"+str(i_robot)]["subgoal0"] = weight_deadl[i_robot]

        t_start_actions = time.perf_counter()

        # --- compute action ---#
        for i_robot in range(nr_robots):
            if state_machine_pandas[i_robot] == 3 or state_machine_pandas[i_robot] == 5:
                action[dof_index[i_robot]:dof_index[i_robot]+dof[i_robot]] = np.zeros(dof[0])
            else:
                arguments_robot = dict(q=q_pandas[i_robot],
                                       qdot=qdot_pandas[i_robot],
                                       x_goal_0=np.array(x_goals["robot_"+str(i_robot)]["subgoal0"]),
                                       x_goal_1=np.array(x_goals["robot_"+str(i_robot)]["subgoal1"]),
                                       x_goal_2=np.array(x_goals["robot_"+str(i_robot)]["subgoal2"]),
                                       weight_goal_0=weight_goals["robot_"+str(i_robot)]["subgoal0"],
                                       weight_goal_1=weight_goals["robot_"+str(i_robot)]["subgoal1"],
                                       weight_goal_2=weight_goals["robot_"+str(i_robot)]["subgoal2"],
                                       angle_goal_1=params.rotation_matrix_pandas[i_robot],
                                       x_obsts=x_dyns_obsts[i_robot],
                                       radius_obsts=params.r_dyns_obsts[i_robot],
                                       constraint_0=params.constraints[i_robot],
                                       radius_body_panda_links=params.radius_body_panda_links,
                                       radius_body_panda_hand=np.array([params.radius_sphere]),
                                       x_obsts_dynamic=x_dyns_obsts[i_robot],
                                       xdot_obsts_dynamic=v_dyns_obsts[i_robot],
                                       xddot_obsts_dynamic=params.a_dyns_obsts[i_robot],
                                       radius_obsts_dynamic=params.r_dyns_obsts[i_robot],
                                       )
                if state_machine_pandas[i_robot] == 2:  # only goal reaching, no obstacle avoidance
                    action[dof_index[i_robot]:dof_index[i_robot]+dof[i_robot]] = planners_grasp[i_robot].compute_action(**arguments_robot)
                else:
                    action[dof_index[i_robot]:dof_index[i_robot] + dof[i_robot]] = planners[i_robot].compute_action(
                            **arguments_robot)
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
                dist_x_r = dist_x - params.r_dyns_obsts[0][k] - params.r_dyns_obsts[1][k]
                if dist_x_r < min_clearance:
                    min_clearance = dist_x_r
                if dist_x_r < 0:
                    print("clearance violation:", dist_x_r)

        # --- for plotting the rollouts and checking if the estimated states match the real states --- #
        if params.ROLLOUTS_PLOTTING == True:
            for df in range(dof[0]):
                state_j[df].append(q_pandas[0][df])
                vel_j[df].append(qdot_pandas[0][df])
                if params.fabrics_mode == "acc":
                    acc_j[df].append(action[df])
                else:
                    acc_j[df].append(0)

            if w % 100 == 0:
                i_robot = 0 # --- currently only for robot 0 plotted! --- #
                print(w)
                # for plotting rollouts
                # Forward Fabrics: Shows the inputs and positions over the horizon
                for df in range(dof[i_robot]):
                    q_N_time_j[df].append(q_robots_N["robot_"+str(i_robot)][df, :])
                    q_dot_N_time_j[df].append(q_dot_robots_N["robot_"+str(i_robot)][df, :])
                    q_ddot_N_time_j[df].append(q_ddot_robots_N["robot_"+str(i_robot)][df, :])

    if params.ROLLOUTS_PLOTTING == True:
        variables_plots = {"state_j": state_j, "vel_j": vel_j, "acc_j": acc_j,
                           "pos_xyz": pos_xyz,
                           "q_N_time_j": q_N_time_j,
                           "qdot_N_time_j": q_dot_N_time_j,
                           "qddot_N_time_j": q_ddot_N_time_j,
                           "q_MPC_N_time_j": q_MPC_N_time_j,
                           "qdot_MPC_N_time_j": q_dot_MPC_N_time_j,
                           "qddot_MPC_N_time_j": q_ddot_MPC_N_time_j,
                           "solver_time": solver_times,
                           "exitflag": exitflag_list[i_robot],
                           "x_goal": x_goals["robot_0"]["subgoal0"],
                            "x_obsts": []}
        Generate_Plots = generate_plots(dof=dof[i_robot], N_steps=n_steps, dt=params.dt, fabrics_mode=params.fabrics_mode, dt_MPC=params.dt)
        Generate_Plots.plot_results(variables_plots, nr_obst=[], r_obst=params.radius_sphere, MPC_LAYER=params.MPC_LAYER)

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

def define_run_panda_example(n_steps=100, render=True):
    with open("examples/configs/panda_config.yaml", "r") as setup_stream:
        setup = yaml.safe_load(setup_stream)
    nr_robots = setup['n_robots']
    random_scene = False
    param = examples.parameters_manipulators.manipulator_parameters(nr_robots=nr_robots, n_obst_per_link=setup['n_obst_per_link'])
    settings = param.define_settings(ROLLOUT_FABRICS=setup['ROLLOUT_FABRICS'],
                                     ROLLOUTS_PLOTTING=setup['ROLLOUTS_PLOTTING'],
                                     STATIC_OR_DYN_FABRICS=setup['STATIC_OR_DYN_FABRICS'],
                                     RESOLVE_DEADLOCKS=setup['RESOLVE_DEADLOCKS'],
                                     ESTIMATE_GOAL=setup['ESTIMATE_GOAL'], N_HORIZON=setup['N_HORIZON'],
                                     n_obst_per_link=setup['n_obst_per_link'])

    #simulation environment:
    simulation_class = create_simulation_manipulators.create_manipulators_simulation(param)
    random_obstacles = simulation_class.create_scene(random_scene=random_scene, n_cubes=param.n_cubes)
    env = simulation_class.initialize_environment(render=render, random_scene=random_scene, obstacles=random_obstacles)

    # forward kinematics:
    link_transforms_list = simulation_class.get_link_transforms()
    utils_class = UtilsKinematics()

    # planners (fabrics)
    planners, planners_grasp, goal_structs = define_planners(params=param)
    fk_dict = utils_class.define_forward_kinematics(planners=planners, collision_links=param.collision_links, collision_links_nrs=param.collision_links_nrs)
    fk_dict_spheres = utils_class.define_symbolic_collision_link_poses(urdf_files=param.urdf_links,
                                                                       collision_links=param.collision_links,
                                                                       sphere_transformations=link_transforms_list,
                                                                       n_obst_per_link=param.n_obst_per_link,
                                                                       mount_transform=param.mount_transform)
    if param.ROLLOUT_FABRICS == True:
        planners_forward = define_rollout_planners(params=param, fk_dict=fk_dict, goal_structs=goal_structs, n_steps=n_steps, planners=planners, nr_robots=nr_robots)
    else:
        planners_forward = None

    res = run_panda_example(params=param, n_steps=n_steps, planners=planners, planners_grasp=planners_grasp,
                            goal_structs=goal_structs, env=env, fk_dict=fk_dict, forwardplanners=planners_forward,
                            fk_dict_spheres=fk_dict_spheres, utils_class=utils_class)
    env.close()
    return res


if __name__ == "__main__":
    n_steps = 7000
    render = True
    res = define_run_panda_example(n_steps=n_steps, render=render)
