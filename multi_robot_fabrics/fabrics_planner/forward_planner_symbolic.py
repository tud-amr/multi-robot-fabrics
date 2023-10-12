import casadi as ca
import numpy as np
import time
import quaternionic
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
import copy
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.gridspec import GridSpec


class ForwardFabricsPlanner(object):
    def __init__(self, params, planners, N_steps, fk_dict,
                 goal_struct_robots, ROLLOUTS_PLOTTING=0):
        self.ROLLOUTS_PLOTTING=ROLLOUTS_PLOTTING
        self.N_horizon = params.N_HORIZON  # control horizon
        self.dt = params.dt  # be careful, this is dt of the planner, not of the simulation
        self.Ts = self.dt
        self.dof = params.dof
        self.nr_robots = len(self.dof)
        self.nr_obsts = params.nr_obsts
        self.radius_obsts = params.radius_obsts
        self.nr_obsts_dyn = params.nr_obsts_dyn
        self.planners = planners
        self.N_steps = N_steps
        self.fk_dict = fk_dict
        self.collision_links_nrs = params.collision_links_nrs
        self.robot_types = params.robot_types
        self.goal_struct_robots = goal_struct_robots
        self.fabrics_mode = params.fabrics_mode
        self.other_robot_static_dynamic = params.STATIC_OR_DYN_FABRICS
        self.r_robots = params.r_robots
        self.r_robots_args = [[] for _ in range(self.nr_robots)]
        self.nr_constraints = params.nr_constraints
        self.n_obst_per_link = params.n_obst_per_link
        self.rotation_matrices_pandas = params.rotation_matrix_pandas
        for i_robot in range(self.nr_robots):
            for z, coll_i in enumerate(self.collision_links_nrs[i_robot]):
                if coll_i > 2:
                    self.r_robots_args[i_robot].append(self.r_robots[i_robot][z])

        # get dynamic obstacles:
        self.a_dyns_obsts = [[] for _ in range(self.nr_robots)]
        self.r_dyns_obsts = [[] for _ in range(self.nr_robots)]
        for i_robot in range(self.nr_robots):
            i_other_robots = [i for i in range(self.nr_robots) if i != i_robot]
            for i_other_robot in i_other_robots:
                for i_sphere in range((len(self.collision_links_nrs[i_other_robot]))):
                    self.a_dyns_obsts[i_robot] = self.a_dyns_obsts[i_robot] + [np.zeros((3,))] * self.n_obst_per_link
                    self.r_dyns_obsts[i_robot] = self.r_dyns_obsts[i_robot] + [
                        self.r_robots[i_other_robot][i_sphere]]  # * self.n_obst_per_link

        self.nr_subgoals = []
        self.q_N_fun = []
        self.q_dot_N_fun = []
        self.q_ddot_N_fun = []
        self.x_obsts_N_fun = [[] for _ in range(self.nr_robots)]
        self.v_obsts_N_fun = [[] for _ in range(self.nr_robots)]
        self.a_obsts_N_fun = [[] for _ in range(self.nr_robots)]
        for i_robot in range(self.nr_robots):
            self.nr_subgoals.append(len(self.goal_struct_robots[i_robot]._config))
            self.q_N_fun.append([[] for k in range(self.dof[0])])
            self.q_dot_N_fun.append([[] for k in range(self.dof[0])])
            self.q_ddot_N_fun.append([[] for k in range(self.dof[0])])

            # if self.ROLLOUTS_PLOTTING:
            #     self.x_obsts_N_fun.append([[] for k in range(self.dof[0])])
            #     self.v_obsts_N_fun.append([[] for k in range(self.dof[0])])
            #     self.a_obsts_N_fun.append([[] for k in range(self.dof[0])])


    def system_step(self, pos, vel, action, i_robot, dt):
        """" System step with integrator to estimate position/velocity along the horizon"""
        if self.fabrics_mode == "acc":
            pos_new = pos + dt * vel + 0.5 * dt ** 2 * np.ones(self.dof[i_robot]) * action
            vel_new = vel + dt * np.ones(self.dof[i_robot]) * action
        else:
            pos_new = pos + dt * action
            vel_new = action
        return pos_new, vel_new

    def forward_kinematics_sym(self, i_robot, q, q_dot, q_ddot):
        """
        Forward kinematics from joint space to task space given the collision links and corresponding forward kinematics
        and jacobian.
        """
        x_robot = [];
        v_robot = [];
        a_robot = []
        for i_link in range(len(self.collision_links_nrs[i_robot])):
            x_dm = self.fk_dict["fk_fun"][i_robot][i_link](q)
            x_robot.append(x_dm)

            v_dm = self.fk_dict["jac_fun"][i_robot][i_link](q) @ q_dot
            v_robot.append(v_dm)

            a_dm = self.fk_dict["jac_fun"][i_robot][i_link](q) @ q_ddot + \
                   self.fk_dict["jac_dot_fun"][i_robot][i_link](q, q_dot) @ q_dot
            a_robot.append(a_dm)
        return x_robot, v_robot, a_robot

    def compute_velocity_average(self, q_dot_robots_N):
        """
        Function used to compute average velocity (symbolically)
        given the joint velocities of both robots along the horizon
        """
        avg_sum = []
        for i_robot in range(self.nr_robots):
            avg_sum_i = 0
            for i in range(self.N_horizon):
                q_dot_N = q_dot_robots_N["robot_" + str(i_robot)][i]
                q_dot_N_squared= q_dot_N**2 #np.sqrt(q_dot_N ** 2)
                q_dot_N_squared_sum = sum(ca.vertsplit(q_dot_N_squared))
                avg_sum_i = avg_sum_i + q_dot_N_squared_sum / (self.N_horizon * self.dof[i_robot])
            avg_sum.append(avg_sum_i)
        return avg_sum

    def forward_multi_fabrics_symbolic(self):
        """
        Computes the symbolic function of the rollouts over the horizon.  This includes:
        - The expression for the state, velocity and acceleration of the joints over the horizon
        - The expression for the average velocity of both robots over the horizon
        """
        q_robots = []
        q_dot_robots = []
        q_ddot_robots = [np.zeros((self.dof[z],)) for z in range(self.nr_robots)]
        x_goals = [[] for z in range(self.nr_robots)]
        weight_goals = [[] for z in range(self.nr_robots)]
        angle_goals_1 = [[] for z in range(self.nr_robots)]
        radius_bodies = [[] for z in range(self.nr_robots)]
        radius_obsts = [[] for z in range(self.nr_robots)]
        radius_obsts_dyn = [[] for z in range(self.nr_robots)]
        constraints = [[] for z in range(self.nr_robots)]
        x_obsts = [[] for z in range(self.nr_robots)]
        xx_robots = [[] for z in range(self.nr_robots)]
        vv_robots = [[] for z in range(self.nr_robots)]
        aa_robots = [[] for z in range(self.nr_robots)]
        q_robots_N = {}
        q_dot_robots_N = {}
        q_ddot_robots_N = {}
        for i_robot in range(self.nr_robots):
            q_robots_N['robot_' + str(i_robot)] = []
            q_dot_robots_N["robot_" + str(i_robot)] = []
            q_ddot_robots_N["robot_" + str(i_robot)] = []
        x_obsts_N = copy.deepcopy(q_robots_N)
        v_obsts_N = copy.deepcopy(q_dot_robots_N)
        a_obsts_N = copy.deepcopy(q_ddot_robots_N)
        action_funs = []
        x_other_obsts_robots = []
        v_other_obsts_robots = []
        a_other_obsts_robots = []
        r_other_obsts_robots = []

        # define parameters symbolically that can be fed real-time to the controller
        for i_robot in range(self.nr_robots):
            for i_subgoal in range(self.nr_subgoals[i_robot]):
                if i_subgoal < 2:
                    x_goals[i_robot].append(ca.SX.sym("x_goal_" + str(i_subgoal) + "_robot" + str(i_robot), 3, 1))
                else:
                    x_goals[i_robot].append(ca.SX.sym("x_goal_" + str(i_subgoal) + "_robot" + str(i_robot), 1, 1))
                weight_goals[i_robot].append(ca.SX.sym('weight_goal_' + str(i_subgoal) + "_robot" + str(i_robot), 1, 1))
            angle_goals_1[i_robot] = ca.SX.sym('angle_goal_1_robot' + str(i_robot), 3, 3)
            for i_coll in self.collision_links_nrs[i_robot]:
                if i_coll > 2:
                    radius_bodies[i_robot].append(
                        ca.SX.sym("radius_body_panda_link" + str(i_coll) + "_robot_" + str(i_robot)))
            for i_obst in range(self.nr_obsts[i_robot]):
                radius_obsts[i_robot].append(ca.SX.sym("radius_obst_" + str(i_obst)))
                x_obsts[i_robot].append(ca.SX.sym("x_obst_" + str(i_obst)))
            for i_obst_dyn in range(self.nr_obsts_dyn[i_robot]):
                radius_obsts_dyn[i_robot].append(
                    ca.SX.sym("radius_obst_" + str(i_obst_dyn) + "_dynamic" + "_robot" + str(i_robot)))
            for i_constraint in range(self.nr_constraints[i_robot]):
                constraints[i_robot].append(
                    ca.SX.sym("constraint_" + str(i_constraint) + "_robot" + str(i_robot), 4, 1))
            q_robots.append(ca.SX.sym("q_robot" + str(i_robot), self.dof[i_robot], 1))
            q_dot_robots.append(ca.SX.sym("q_robot" + str(i_robot), self.dof[i_robot], 1))

        args_action = [[] for _ in range(self.nr_robots)]

        for i_robot in range(self.nr_robots):
            # q_robots.append(self.planners[i_robot].variables.position_variable())
            # q_dot_robots.append(self.planners[i_robot].variables.velocity_variable())
            input_keys = self.planners[0]._funs._input_keys
            action_funs.append(self.planners[i_robot]._funs._function)

        q_cur = copy.deepcopy(q_robots)
        q_dot_cur = copy.deepcopy(q_dot_robots)

        for k in range(self.N_horizon):
            for i_robot in range(self.nr_robots):
                if self.fabrics_mode == "acc":
                    q_robots[i_robot], q_dot_robots[i_robot] = self.system_step(pos=q_robots[i_robot],
                                                                                vel=q_dot_robots[i_robot],
                                                                                action=q_ddot_robots[i_robot],
                                                                                i_robot=i_robot, dt=self.dt)
                elif self.fabrics_mode == "vel":
                    q_robots[i_robot], q_dot_robots[i_robot] = self.system_step(pos=q_robots[i_robot],
                                                                                vel=[],
                                                                                action=q_dot_robots[i_robot],
                                                                                i_robot=i_robot, dt=self.dt)
                q_ddot_robots[i_robot] = np.zeros((self.dof[i_robot]))
                xx_robots[i_robot], vv_robots[i_robot], aa_robots[i_robot] = self.forward_kinematics_sym(i_robot,
                                                                                                         q_robots[
                                                                                                             i_robot],
                                                                                                         q_dot_robots[
                                                                                                             i_robot],
                                                                                                         q_ddot_robots[
                                                                                                             i_robot])

            for i_robot in range(self.nr_robots):
                i_other_robots = [i for i in range(self.nr_robots) if i != i_robot]
                for i_others in i_other_robots:
                    x_other_obsts = xx_robots[i_others]
                    if self.other_robot_static_dynamic == 0:
                        v_other_obsts = [np.zeros(3)] * len(vv_robots[i_others])
                        a_other_obsts = [np.zeros(3)] * len(aa_robots[i_others])
                    else:
                        v_other_obsts = vv_robots[i_others]
                        a_other_obsts = aa_robots[i_others]
                    r_other_obsts = self.r_robots[i_others]
                    x_other_obsts_robots = x_other_obsts_robots + x_other_obsts
                    v_other_obsts_robots = v_other_obsts_robots + v_other_obsts
                    a_other_obsts_robots = a_other_obsts_robots + a_other_obsts
                    r_other_obsts_robots = r_other_obsts_robots + r_other_obsts

                args_action[i_robot] = [angle_goals_1[i_robot], *constraints[i_robot],
                                        q_robots[i_robot], q_dot_robots[i_robot],
                                        *radius_bodies[i_robot], *radius_obsts[i_robot], *r_other_obsts_robots,
                                        *weight_goals[i_robot], *x_goals[i_robot],
                                        *x_obsts[i_robot], *x_other_obsts_robots, *a_other_obsts_robots,
                                        *v_other_obsts_robots]
                q_dot_robots[i_robot] = action_funs[i_robot](*args_action[i_robot])

                # store results in list over the horizon
                # for df in range(self.dof[i_robot]):
                q_robots_N['robot_' + str(i_robot)].append(q_robots[i_robot])
                q_dot_robots_N['robot_' + str(i_robot)].append(q_dot_robots[i_robot])
                q_ddot_robots_N['robot_' + str(i_robot)].append(q_ddot_robots[i_robot])

                if self.ROLLOUTS_PLOTTING:
                    x_obsts_N['robot_' + str(i_robot)].append(x_other_obsts_robots)
                    v_obsts_N['robot_' + str(i_robot)].append(v_other_obsts_robots)
                    a_obsts_N['robot_' + str(i_robot)].append(a_other_obsts_robots)

                x_other_obsts_robots = []
                v_other_obsts_robots = []
                a_other_obsts_robots = []
                r_other_obsts_robots = []

        # ---- construct argument vector ----- #
        args_action_funs = []
        for i_robot in range(self.nr_robots):
            if self.nr_obsts_dyn[i_robot] > 0:  # if dynamic obstacles present
                args_action_fun = []
                for i_rob in range(self.nr_robots):
                    args_action_i = [angle_goals_1[i_rob],
                                     *constraints[i_rob],
                                     q_cur[i_rob], q_dot_cur[i_rob],
                                     *weight_goals[i_rob], *x_goals[i_rob],
                                     *x_obsts[i_rob], *radius_obsts[i_rob],
                                     *radius_bodies[i_rob],
                                     *radius_obsts_dyn[i_rob]]
                    args_action_fun = args_action_fun + args_action_i
            else:  # if no dynamic obstacles present
                args_action_fun = [angle_goals_1[i_robot],
                                   constraints[i_robot],
                                   q_cur[i_robot], q_dot_cur[i_robot],
                                   *weight_goals[i_robot], *x_goals[i_robot],
                                   *x_obsts[i_robot], *radius_obsts[i_robot], *radius_bodies[i_robot],
                                   *radius_obsts_dyn[i_robot]]

            args_action_funs.append(args_action_fun)


            self.q_N_fun[i_robot] = ca.Function("q_robot_" + str(i_robot) + "_N", args_action_fun,
                                                q_robots_N["robot_" + str(i_robot)])
            self.q_dot_N_fun[i_robot] = ca.Function("q_dot_robot_" + str(i_robot) + "_N", args_action_fun,
                                                    q_dot_robots_N["robot_" + str(i_robot)])
            self.q_ddot_N_fun[i_robot] = ca.Function("q_ddot_robot_" + str(i_robot) + "_N", args_action_fun,
                                                     q_ddot_robots_N["robot_" + str(i_robot)])

            if self.ROLLOUTS_PLOTTING:
                self.x_obsts_N_fun[i_robot].append([ca.Function("q_robot_" + str(i_robot) + "_N", args_action_fun,
                                                              x_obsts_N["robot_" + str(i_robot)][k]) for k in range(self.N_horizon)])
                self.v_obsts_N_fun[i_robot].append([ca.Function("q_dot_robot_" + str(i_robot) + "_N", args_action_fun,
                                                              v_obsts_N["robot_" + str(i_robot)][k]) for k in range(self.N_horizon)])
                self.a_obsts_N_fun[i_robot].append([ca.Function("q_ddot_robot_" + str(i_robot) + "_N", args_action_fun,
                                                              a_obsts_N["robot_" + str(i_robot)][k]) for k in range(self.N_horizon)])

        avg_vel_sym = self.compute_velocity_average(q_dot_robots_N)
        if self.nr_obsts_dyn[0] > 0:
            self.avg_vel_fun = ca.Function("avg_vel", [*args_action_funs[0]], avg_vel_sym)
        else:
            self.avg_vel_fun = ca.Function("avg_vel", [*args_action_funs[0], *args_action_funs[1]], [avg_vel_sym])
        return {}

    def get_velocity_rollouts(self, inputs_action):
        """
        Numerical value of the average velocity of the rollouts over the horizon.
        Only this value is needed for deadlock detection (not the full rollouts which makes it faster).
        """
        arguments = []
        for i_robot in range(self.nr_robots):
            if i_robot == 0:
                i_other_robot = 1
            else:
                i_other_robot = 0
            other_robots = []
            for i_robot_other in range(self.nr_robots):
                if i_robot != i_robot_other:
                    other_robots.append(i_robot_other)
            arguments.append(self.rotation_matrices_pandas[i_robot])
            for i_constraints in range(self.nr_constraints[i_robot]):
                arguments.append(inputs_action["constraints"][i_robot])
            arguments.append(inputs_action["q_robots"][i_robot])
            arguments.append(inputs_action["q_dot_robots"][i_robot])
            for i_subgoal in range(self.nr_subgoals[i_robot]):
                arguments.append(inputs_action["weight_goals" + str(i_subgoal)][i_robot])
            for i_subgoal in range(self.nr_subgoals[i_robot]):
                arguments.append(inputs_action["x_goals" + str(i_subgoal)][i_robot])
            for i_obst in range(self.nr_obsts[i_robot]):
                arguments.append(inputs_action["x_obsts"][i_robot][i_obst])
            for i_obst in range(self.nr_obsts[i_robot]):  #static obstacles
                arguments.append(self.radius_obsts[i_robot][i_obst])
            for r_body_robot in self.r_robots_args[i_robot]:  # radius bodies
                arguments.append(r_body_robot)
            for r_other_robot in self.r_dyns_obsts[i_other_robot]:  # radius dynamic obstacles
                arguments.append(r_other_robot)
        avg_vel = []
        for i_robot in range(self.nr_robots):
            time_start_symbolic = time.perf_counter()
            avg_vel.append(self.avg_vel_fun(*arguments)[i_robot].full()[0])
            time_diff = time.perf_counter() - time_start_symbolic
            # print("time it takes to just call the symbolic function:", time_diff)
        return avg_vel

    def rollouts_numerical(self, inputs_action):
        """
        Uses the symbolic function of the rollouts to get the numerical values of the state, velocity and acceleration
        of the rollouts over the horizon given the input parameters.
        Input = dictionary with:
        angle_goals1, q_robots, q_dot_robots, weight_goals, x_goals, x_obsts, radius_obsts
        Output = list of joint state, vel, acc over N.
        """
        q_num_robots_N = {"robot_0": [], "robot_1": []}
        q_dot_num_robots_N = {"robot_0": [], "robot_1": []}
        q_ddot_num_robots_N = {"robot_0": [], "robot_1": []}

        i_robot = 0
        if self.nr_obsts_dyn[i_robot] > 0:
            arguments = []
            for i_robot in range(self.nr_robots):
                if i_robot == 0:
                    i_other_robot = 1
                else:
                    i_other_robot = 0
                other_robots = []
                for i_robot_other in range(self.nr_robots):
                    if i_robot != i_robot_other:
                        other_robots.append(i_robot_other)
                arguments.append(self.rotation_matrices_pandas[i_robot])
                for i_constraints in range(self.nr_constraints[i_robot]):
                    arguments.append(inputs_action["constraints"][i_robot])
                arguments.append(inputs_action["q_robots"][i_robot])
                arguments.append(inputs_action["q_dot_robots"][i_robot])
                for i_subgoal in range(self.nr_subgoals[i_robot]):
                    arguments.append(inputs_action["weight_goals" + str(i_subgoal)][i_robot])
                for i_subgoal in range(self.nr_subgoals[i_robot]):
                    arguments.append(inputs_action["x_goals" + str(i_subgoal)][i_robot])
                for i_obst in range(self.nr_obsts[i_robot]):
                    arguments.append(inputs_action["x_obsts"][i_robot][i_obst])
                for i_obst in range(self.nr_obsts[i_robot]):  # static obstacles
                    arguments.append(self.radius_obsts[i_robot][i_obst])
                for r_body_robot in self.r_robots_args[i_robot]:  # radius bodies
                    arguments.append(r_body_robot)
                for r_other_robot in self.r_dyns_obsts[i_other_robot]:  # radius dynamic obstacles
                    arguments.append(r_other_robot)

            for i_robot in range(self.nr_robots):
                solution_dm = self.q_N_fun[i_robot](*arguments)
                solution = np.array([solution_i.full() for solution_i in list(solution_dm)]).transpose()[0]
                q_num_robots_N["robot_" + str(i_robot)].append(solution)

                solution_dm = self.q_dot_N_fun[i_robot](*arguments)
                solution = np.array([solution_i.full() for solution_i in list(solution_dm)]).transpose()[0]
                q_dot_num_robots_N["robot_" + str(i_robot)].append(solution)

                solution_dm = self.q_ddot_N_fun[i_robot](*arguments)
                solution = np.array([solution_i.full() for solution_i in list(solution_dm)]).transpose()[0]
                q_ddot_num_robots_N["robot_" + str(i_robot)].append(solution)

        else:
            for i_robot in range(self.nr_robots):
                arguments = []
                arguments.append(inputs_action["angle_goals1"][i_robot])
                arguments.append(inputs_action["q_robots"][i_robot])
                arguments.append(inputs_action["q_dot_robots"][i_robot])
                for i_subgoal in range(self.nr_subgoals[i_robot]):
                    arguments.append(inputs_action["weight_goals" + str(i_subgoal)][i_robot])
                for i_subgoal in range(self.nr_subgoals[i_robot]):
                    arguments.append(inputs_action["x_goals" + str(i_subgoal)][i_robot])
                for i_obst in range(self.nr_obsts[i_robot]):
                    arguments.append(inputs_action["x_obsts"][i_obst])
                for i_obst in range(self.nr_obsts[i_robot]):
                    arguments.append(inputs_action["r_obsts"][i_obst])
                for r_other_robot in self.r_robots_args[i_robot]:
                    arguments.append(r_other_robot)

                for df in range(self.dof[i_robot]):
                    solution_dm = self.q_N_fun[i_robot][df](*arguments)
                    solution = np.array([solution_i.full()[0] for solution_i in list(solution_dm)]).transpose()[0]
                    q_num_robots_N["robot_" + str(i_robot)].append(solution)

                    solution_dm = self.q_dot_N_fun[i_robot][df](*arguments)
                    solution = np.array([solution_i.full()[0] for solution_i in list(solution_dm)]).transpose()[0]
                    q_dot_num_robots_N["robot_" + str(i_robot)].append(solution)

                    solution_dm = self.q_ddot_N_fun[i_robot][df](*arguments)
                    solution = np.array([solution_i.full()[0] for solution_i in list(solution_dm)]).transpose()[0]
                    q_ddot_num_robots_N["robot_" + str(i_robot)].append(solution)

        return q_num_robots_N, q_dot_num_robots_N, q_ddot_num_robots_N

    def rollouts_numerical_obstacles(self, inputs_action):
        """
        Uses the symbolic function of the rollouts to get the numerical values of the state, velocity and acceleration
        of the rollouts over the horizon given the input parameters.
        Input = dictionary with:
        angle_goals1, q_robots, q_dot_robots, weight_goals, x_goals, x_obsts, radius_obsts
        Output = list of joint state, vel, acc over N.
        """
        x_num_robots_N = {"robot_0": [], "robot_1": []}
        v_num_robots_N = {"robot_0": [], "robot_1": []}
        a_num_robots_N = {"robot_0": [], "robot_1": []}

        i_robot = 0
        if self.nr_obsts_dyn[i_robot] > 0:
            arguments = []
            for i_robot in range(self.nr_robots):
                if i_robot == 0:
                    i_other_robot = 1
                else:
                    i_other_robot = 0
                other_robots = []
                for i_robot_other in range(self.nr_robots):
                    if i_robot != i_robot_other:
                        other_robots.append(i_robot_other)
                arguments.append(self.rotation_matrices_pandas[i_robot])
                for i_constraints in range(self.nr_constraints[i_robot]):
                    arguments.append(inputs_action["constraints"][i_robot])
                arguments.append(inputs_action["q_robots"][i_robot])
                arguments.append(inputs_action["q_dot_robots"][i_robot])
                for i_subgoal in range(self.nr_subgoals[i_robot]):
                    arguments.append(inputs_action["weight_goals" + str(i_subgoal)][i_robot])
                for i_subgoal in range(self.nr_subgoals[i_robot]):
                    arguments.append(inputs_action["x_goals" + str(i_subgoal)][i_robot])
                for i_obst in range(self.nr_obsts[i_robot]):
                    arguments.append(inputs_action["x_obsts"][i_robot][i_obst])
                for i_obst in range(self.nr_obsts[i_robot]):  # static obstacles
                    arguments.append(self.radius_obsts[i_robot][i_obst])
                for r_body_robot in self.r_robots_args[i_robot]:  # radius bodies
                    arguments.append(r_body_robot)
                for r_other_robot in self.r_dyns_obsts[i_other_robot]:  # radius dynamic obstacles
                    arguments.append(r_other_robot)

            for i_robot in range(self.nr_robots):
                for k in range(self.N_horizon):
                    solution_dm = self.x_obsts_N_fun[i_robot][0][k](*arguments)
                    solution = np.array([solution_i.full() for solution_i in list(solution_dm)]).transpose()[0]
                    x_num_robots_N["robot_" + str(i_robot)].append(solution)

                    solution_dm = self.v_obsts_N_fun[i_robot][0][k](*arguments)
                    solution = np.array([solution_i.full() for solution_i in list(solution_dm)]).transpose()[0]
                    v_num_robots_N["robot_" + str(i_robot)].append(solution)

                    solution_dm = self.a_obsts_N_fun[i_robot][0][k](*arguments)
                    solution = np.array([solution_i.full() for solution_i in list(solution_dm)]).transpose()[0]
                    a_num_robots_N["robot_" + str(i_robot)].append(solution)

        else:
            for i_robot in range(self.nr_robots):
                arguments = []
                arguments.append(inputs_action["angle_goals1"][i_robot])
                arguments.append(inputs_action["q_robots"][i_robot])
                arguments.append(inputs_action["q_dot_robots"][i_robot])
                for i_subgoal in range(self.nr_subgoals[i_robot]):
                    arguments.append(inputs_action["weight_goals" + str(i_subgoal)][i_robot])
                for i_subgoal in range(self.nr_subgoals[i_robot]):
                    arguments.append(inputs_action["x_goals" + str(i_subgoal)][i_robot])
                for i_obst in range(self.nr_obsts[i_robot]):
                    arguments.append(inputs_action["x_obsts"][i_obst])
                for i_obst in range(self.nr_obsts[i_robot]):
                    arguments.append(inputs_action["r_obsts"][i_obst])
                for r_other_robot in self.r_robots_args[i_robot]:
                    arguments.append(r_other_robot)

                for i_robot in range(self.nr_robots):
                    for k in range(self.N_horizon):
                        solution_dm = self.x_obsts_N_fun[i_robot][0][k](*arguments)
                        solution = np.array([solution_i.full() for solution_i in list(solution_dm)]).transpose()[0]
                        x_num_robots_N["robot_" + str(i_robot)].append(solution)

                        solution_dm = self.v_obsts_N_fun[i_robot][0][k](*arguments)
                        solution = np.array([solution_i.full() for solution_i in list(solution_dm)]).transpose()[0]
                        v_num_robots_N["robot_" + str(i_robot)].append(solution)

                        solution_dm = self.a_obsts_N_fun[i_robot][0][k](*arguments)
                        solution = np.array([solution_i.full() for solution_i in list(solution_dm)]).transpose()[0]
                        a_num_robots_N["robot_" + str(i_robot)].append(solution)

        return x_num_robots_N, v_num_robots_N, a_num_robots_N

    def plot_results(self, variables_plots):
        time_x = np.linspace(0, self.N_steps * 0.01, self.N_steps)
        n_steps_lim = int(self.N_steps / 100)  # don't want to plot all steps
        i_robot = 0

        # plotting
        colorscheme = ['r', 'b', 'm', 'g', 'k', 'y', 'deeppink']

        if n_steps_lim > 2:
            n_plots_N = 3
        else:
            n_plots_N = n_steps_lim

        fig2, ax2 = plt.subplots(n_plots_N, 3, figsize=(20, 15))
        # Plots over N
        for i in range(n_plots_N):
            i_dist = n_steps_lim / n_plots_N
            i_plt = int(np.floor(i_dist * i))
            time_0_N = i_plt * 100
            plt.gca().set_prop_cycle(None)
            for df in range(self.dof[i_robot]):
                ax2[i, 1].plot(variables_plots["q_dot_N_time_j"][df][i_plt], color=colorscheme[df])
                ax2[i, 1].plot(variables_plots["vel_j"][df][time_0_N:time_0_N + self.N_horizon], 'x',
                               color=colorscheme[df])

                ax2[i, 2].plot(variables_plots["action_N_time_j"][df][i_plt], color=colorscheme[df])
                ax2[i, 2].plot(variables_plots["acc_j"][df][time_0_N:time_0_N + self.N_horizon], 'x',
                               color=colorscheme[df])

                ax2[i, 0].plot(variables_plots["state_N_time_j"][df][i_plt], color=colorscheme[df])
                ax2[i, 0].plot(variables_plots["state_j"][df][time_0_N:time_0_N + self.N_horizon], 'x',
                               color=colorscheme[df])
            ax2[i, 2].set_title("Action ($\ddot{q}$) over N, time=" + str(i_plt * 100 * self.dt));
            ax2[i, 2].set_xlabel('$i=1:N$')
            ax2[i, 2].set_ylabel('$\ddot{q}')
            ax2[i, 1].set_title("Velocities ($\dot{q}$) over N, time=" + str(i_plt * 100 * self.dt));
            ax2[i, 0].set_title("Joint pos ($q$) over N, time=" + str(i_plt * 100 * self.dt));
            ax2[i, 0].set_xlabel('$i=1:N$');
            ax2[i, 0].set_ylabel('$q$')
            ax2[i, 0].grid();
            ax2[i, 1].grid()
            ax2[i, 2].grid()
        plt.show()