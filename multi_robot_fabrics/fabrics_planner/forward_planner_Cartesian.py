import casadi as ca
import numpy as np
import time
import copy
import quaternionic
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

class FabricsRollouts(object):
    """
    This class is used to rollout fabrics. this can either be done:
    1) by constructing a symbolic struct beforehand, which is captured into a casadi function for which numerical values
    can be implemented.
    2) By doing the numerical rollouts during the executions.
    The first method is slightly faster, but the difference is approximately factor a half.

    !!!!!!IMPORTANT!!!!!
    Compared to the file: forward_planner_symbolic: obstacles positions and velocitiesare not dependent on forward kinematics,
    but forward simulated using a constant velocity in CARTESIAN space.
    """
    def __init__(self, N, dt, nx, nu, dof, nr_obsts, bool_ring, nr_obsts_dyn=0, v_obsts_dyn=[], fabrics_mode="acc",
                 collision_links_nrs=[7], nr_constraints=0, radius_sphere=0.08, constraints=None, nr_goals=3):
        self.N = N    # control horizon
        self.dt = dt  # be careful, this is dt of the planner, not of the simulation
        self.Ts = self.dt
        self.nx = nx  # state dimension
        self.nu = nu  # input dimension
        self.dof = dof
        self.ring = bool_ring
        self.nr_obsts = nr_obsts
        self.nr_goals = nr_goals
        self.nr_obsts_dyn=nr_obsts_dyn
        self.v_obsts_dyn = v_obsts_dyn
        self.a_obsts_dyn = [np.zeros((3, ))]*len(v_obsts_dyn)
        self.fabrics_mode = fabrics_mode
        self.collision_links_nrs = collision_links_nrs
        self.nr_constraints = nr_constraints
        self.radius_sphere = radius_sphere
        self.robot_type = "panda"
        self.rotation_matrix_panda = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.radius_obsts_dyn = []
        self.radius_obsts = []
        self.constraints=constraints

        self.radius_body_panda_links = {}
        for j, col_link in enumerate(self.collision_links_nrs):
            if self.robot_type == "panda" and col_link > 2:
                # radii on ego body are not considered for obstacle avoidance until link 2, since they are fixed
                self.radius_body_panda_links[str(col_link)] = np.array(self.radius_sphere)

    def preset_radii(self, ob_robot):
        """
        predefine the radii of the obstacles, since they are constant over time.
        """
        if self.nr_obsts+self.nr_obsts_dyn>0:
            first_index = list(ob_robot['FullSensor']['obstacles'].keys())[0]
            self.radius_obsts = [
                ob_robot['FullSensor']['obstacles'][first_index+i]["size"] for i in range(self.nr_obsts)
            ]
            if self.ring:
                self.radius_obsts_dyn = [
                    ob_robot['FullSensor']['obstacles'][first_index+self.nr_obsts + i]["size"] for i in range(self.nr_obsts_dyn)
                ]
            else:
                self.radius_obsts_dyn = [
                    ob_robot['FullSensor']['obstacles'][first_index+self.nr_obsts + i]["size"] for i in range(self.nr_obsts_dyn)
                ]
        else:
            self.radius_obsts_dyn = []
            self.radius_obsts = []

    def preset_radii_obsts_dyn(self, radii_obst_dyn):
        self.radius_obsts_dyn = radii_obst_dyn

    def reset_v_obsts_dyn(self, v_obsts_dyn):
        self.v_obsts_dyn = v_obsts_dyn

    def system_step(self, pos, vel, input, dt:float, fabrics_mode="vel"):
        """
        A first-order system step of any dimensional system.
        """
        dimension = pos.shape[0]
        if fabrics_mode == "acc":
            pos_new = pos + dt*vel + 0.5*dt**2*np.ones(dimension) * input
            vel_new = vel + dt*np.ones(dimension) * input
        elif fabrics_mode == "vel":
            pos_new = pos + dt*np.ones(dimension)*input
            vel_new = input
        else:
            pos_new = []
            vel_new = []
            print("nonexisting fabrics mode inserted, should be vel or acc")
        return pos_new, vel_new

    def get_x_obsts(self, ob_robot) -> list:
        """
        The cartesian positions of the obstacles retrieved from the environment.
        """
        first_index = list(ob_robot['FullSensor']['obstacles'].keys())[0]
        x_obsts = [
            ob_robot['FullSensor']['obstacles'][first_index+i]["position"] for i in range(self.nr_obsts)
        ]
        return x_obsts

    def get_x_obsts_dyn_current(self, ob_robot)-> list:
        """
        The cartesian positions of the dynamic obstacles from the environment
        """
        first_index = list(ob_robot['FullSensor']['obstacles'].keys())[0]
        if self.ring:
            x_obsts_dyn = [
                ob_robot['FullSensor']['obstacles'][first_index+self.nr_obsts + i]["position"] for i in range(self.nr_obsts_dyn)
            ]
        else:
            x_obsts_dyn = [
                ob_robot['FullSensor']['obstacles'][first_index+self.nr_obsts + i]["position"] for i in range(self.nr_obsts_dyn)
            ]
        return x_obsts_dyn

    def get_goal_x_weight(self, ob_robot, goal) -> (list, list):
        """
        Retrieve the goal and goal weight from the environment
        """
        first_index = list(ob_robot['FullSensor']['goals'].keys())[0]
        x_goals = [
            ob_robot['FullSensor']['goals'][first_index+i]["position"] for i in range(self.nr_goals)
        ]
        weight_goals = [
            goal.sub_goals()[i].weight() for i in range(self.nr_goals)
        ]
        return x_goals, weight_goals

    def get_action(self, planner, pos, vel, x_obsts:list, x_obsts_dyn:list, x_goals:list, weight_goals:list):
        """
        Returns the action of the fabrics planner given the current:
        - joint position of the robot
        - joint velocity of the robot
        - cartesian position of the obstacles
        - cartesian position of the dynamic obstacles
        - cartesian position of the goals
        - weight of the goals
        """
        #comment: this would have been nicer if integrated into fabrics package, now append some zeros
        diff_length = 3 - self.nr_goals
        if diff_length>0:
            for i in range(diff_length):
                x_goals.append(0)
                weight_goals.append(0)

        if self.ring:
            action = planner.compute_action(
                q=pos,
                qdot=vel,
                x_obsts=x_obsts,
                radius_obsts=self.radius_obsts,
                angle_goal_1=self.rotation_matrix_panda,
                x_goal_0=x_goals[0],
                x_goal_1=x_goals[1],
                x_goal_2=x_goals[2],
                weight_goal_0=weight_goals[0],
                weight_goal_1=weight_goals[1],
                weight_goal_2=weight_goals[2],
                x_obsts_dynamic = x_obsts_dyn,
                xdot_obsts_dynamic = self.v_obsts_dyn,
                xddot_obsts_dynamic = [np.array([0.0, 0.0, 0.0])]*self.nr_obsts_dyn,
                radius_obsts_dynamic = self.radius_obsts_dyn,
                radius_body_panda_links=self.radius_body_panda_links,
                radius_body_panda_hand=np.array([0.08]),
                constraint_0 = self.constraints
            )
        else:
            action = planner.compute_action(
                q=pos,
                qdot=vel,
                angle_goal_1=self.rotation_matrix_panda,
                x_goal_0=x_goals[0],
                x_goal_1=x_goals[1],
                x_goal_2=x_goals[2],
                weight_goal_0=weight_goals[0],
                weight_goal_1=weight_goals[1],
                weight_goal_2=weight_goals[2],
                x_obsts = x_obsts,
                radius_obsts=self.radius_obsts,
                x_obsts_dynamic=x_obsts_dyn,
                xdot_obsts_dynamic=self.v_obsts_dyn,
                xddot_obsts_dynamic=[np.array([0.0, 0.0, 0.0])]*self.nr_obsts_dyn,
                radius_obsts_dynamic=self.radius_obsts_dyn,
                radius_body_panda_links=self.radius_body_panda_links,
                radius_body_panda_hand=np.array([0.08]),
                constraint_0=self.constraints
            )
        return action


############################### numerical forward fabrics ##########################################
    def get_x_obsts_dyn_N(self, x_obsts_dyn):
        """"
        Returns the positions of the dynamic obstacles along the control horizon.
        from k=0 to k=N (so in python N+1 steps)
        """
        x_obsts_dyn_0 = np.stack(x_obsts_dyn).transpose()
        v_obsts_dyn = np.stack(self.v_obsts_dyn).transpose()
        x_obsts_dyn_N = [np.array([]) for _ in range(self.N+1)]
        x_obsts_dyn_N[0] = x_obsts_dyn_0
        Ts_3d = self.Ts*np.ones((3, self.nr_obsts_dyn))
        for i in range(self.N):
            # for i_obst in range(self.nr_obst_dyn):
             x_obsts_dyn_N[i+1] = x_obsts_dyn_N[i]+v_obsts_dyn*Ts_3d

        x_obsts_dyn_N_list = [[[] for _ in range(self.nr_obsts_dyn)] for _ in range(self.N)]
        x_obsts_dyn_N_list[0] = np.array(x_obsts_dyn)
        Ts_3d = np.array([self.Ts, self.Ts, self.Ts])
        for i in range(self.N - 1):
            for i_obst in range(self.nr_obsts_dyn):
                x_obsts_dyn_N_list[i + 1][i_obst] = x_obsts_dyn_N_list[i][i_obst] + self.v_obsts_dyn[i_obst] * Ts_3d

        return x_obsts_dyn_N, x_obsts_dyn_N_list

    def forward_fabrics(self, planner, pos_k, vel_k, ob_robot, goal, x_obsts_dyn_0 = None, x_goals_struct=None, weight_goals_struct=None):
        """
        Forward simulates fabrics over the horizon numerically
        Outputs a list of joint positions, velocities and accelerations along the horizon.

        The inputs x_obsts_dyn at N=0, goal and weights struct can be given. If not, they will be constructed from the
        environment description.
        """
        u_k = []
        fabr_forw = {"U_stacked":[ [] for _ in range(self.dof) ], "X_stacked":[ [] for _ in range(self.dof) ]}
        q_stacked = []
        qdot_stacked = []
        qddot_stacked = []

        for i in range(self.N):
            #construct inputs from ob_robot struct:
            if self.nr_obsts > 0:
                x_obsts = self.get_x_obsts(ob_robot)
            else:
                x_obsts = []
            if self.nr_obsts_dyn > 0:
                if x_obsts_dyn_0 == None:
                    x_obsts_dyn = self.get_x_obsts_dyn_current(ob_robot)
                else:
                    x_obsts_dyn = x_obsts_dyn_0
                x_obsts_dyn_N, x_obsts_dyn_N_list = self.get_x_obsts_dyn_N(x_obsts_dyn)
            else:
                x_obsts_dyn_N_list = [[] for _ in range(self.N)]

            if x_goals_struct == None:
                x_goals, weight_goals = self.get_goal_x_weight(ob_robot, goal)
            else:
                x_goals = list(x_goals_struct.values())
                weight_goals = list(weight_goals_struct.values())

            #input using current state
            u_k[0:self.dof] = self.get_action(planner, pos_k, vel_k, x_obsts=x_obsts, x_obsts_dyn=x_obsts_dyn_N_list[i], x_goals=x_goals, weight_goals=weight_goals)

            #next state
            [pos_k, vel_k] = self.system_step(pos_k, vel_k, u_k[0:self.dof], dt=self.dt, fabrics_mode=self.fabrics_mode)
            if self.fabrics_mode == "acc":
                state_k = np.append(pos_k, vel_k)
            elif self.fabrics_mode == "vel":
                state_k = pos_k
            else:
                state_k = pos_k
                print("nonexisting fabrics mode inserted, should be vel or acc")

            if self.fabrics_mode == "acc":
                qddot_stacked.append(u_k.copy())
                qdot_stacked.append(vel_k.copy())
            else:
                qdot_stacked.append(u_k.copy())
            q_stacked.append(pos_k.copy())

        return q_stacked, qdot_stacked, qddot_stacked

############################################ symbolic forward fabrics ############################################
    def compute_velocity_average(self, q_dot_N):
        """
        Function used to compute average velocity (symbolically)
        given the joint velocities of both robots along the horizon
        """
        avg_sum = []
        avg_sum_i = 0
        for i in range(self.N):
            q_dot_N_squared = q_dot_N[i] ** 2  # np.sqrt(q_dot_N ** 2)
            q_dot_N_squared_sum = sum(ca.vertsplit(q_dot_N_squared))
            avg_sum_i = avg_sum_i + q_dot_N_squared_sum / (self.N * self.dof)
        avg_sum.append(avg_sum_i)
        return avg_sum

    def get_name_list_variables(self)-> dict:
        """
        For readability, a list of names is constructed to serve as input and output names for the casadi functions.
        This makes debugging/life easier.
        """
        name_dict = {"x_goals":[], "weight_goals":[], "angle_goals_1":[], "radius_bodies":[], "radius_obsts":[],
                     "x_obsts":[], "radius_obsts_dyn":[], "x_obsts_dyn":[], "v_obsts_dyn":[], "a_obsts_dyn":[],
                     "constraints":[], "q_robots":[], "q_dot_robots":[],
                     "q_robots_N":[], "q_dot_robots_N":[], "q_ddot_robots_N":[], "x_obsts_dyn_N":[]}

        for i_subgoal in range(self.nr_subgoals):
            if i_subgoal < 2:
                name_dict["x_goals"].append("x_goal_" + str(i_subgoal))
            else:
                name_dict["x_goals"].append("x_goal_" + str(i_subgoal))
            name_dict["weight_goals"].append('weight_goal_' + str(i_subgoal))
        if self.nr_subgoals>1:
            name_dict["angle_goals_1"] = ["angle_goal_1"]
        else:
            name_dict["angle_goals_1"] = []

        # radius of the ego-robot bodies: #
        if self.nr_obsts + self.nr_obsts_dyn > 0:
            for i_coll in self.collision_links_nrs:
                if i_coll > 2:
                    name_dict["radius_bodies"].append("radius_body_panda_link" + str(i_coll))

        # static obstacles: #
        for i_obst in range(self.nr_obsts):
            name_dict["radius_obsts"].append("radius_obst_" + str(i_obst))
            name_dict["x_obsts"].append("x_obst_" + str(i_obst))

        # dynamic obstacles: start positions: #
        for i_obst_dyn in range(self.nr_obsts_dyn):
            name_dict["radius_obsts_dyn"].append("radius_obst_" + str(i_obst_dyn) + "_dynamic")
            name_dict["x_obsts_dyn"].append("x_obst_" + str(i_obst_dyn))
            name_dict["v_obsts_dyn"].append("v_obst_" + str(i_obst_dyn))
            name_dict["a_obsts_dyn"].append("a_obst_" + str(i_obst_dyn))

        # plane constraints: #
        for i_constraint in range(self.nr_constraints):
            name_dict["constraints"].append("constraint_" + str(i_constraint))

        # state (position and velocity): #
        name_dict["q_robots"] = ["q"]
        name_dict["q_dot_robots"] = ["q_dot"]

        # ---- output names ---- #
        for k in range(self.N):
            name_dict["q_robots_N"].append("q_robots_k_"+str(k))
            name_dict["q_dot_robots_N"].append("q_dot_robots_k_"+str(k))
            name_dict["q_ddot_robots_N"].append("q_ddot_robots_k_"+str(k))

            x_obsts_dyn_k = ["x_obsts_dyn_k_" + str(k) + "_obst_" + str(i) for i in range(self.nr_obsts_dyn)]
            name_dict["x_obsts_dyn_N"].append(x_obsts_dyn_k)
        return name_dict

    def symbolic_forward_fabrics(self, planner, goal_struct):
        """
        Construct casadi functions with fabrics as action of the joint positions, velocities and accelerations along the horizon
        Additionally: the dynamic obstacles are forward simulated using a constant velocity along the horizon.
        """
        q_ddot_robots = np.zeros((self.dof))
        x_goals = []
        weight_goals = []
        radius_bodies = []
        radius_obsts = []
        radius_obsts_dyn = []
        constraints = []
        x_obsts = []
        x_obsts_dyn = []
        v_obsts_dyn = []
        a_obsts_dyn = []
        q_robots_N = [[] for _ in range(self.N)]
        q_dot_robots_N = [[] for _ in range(self.N)]
        q_ddot_robots_N= [[] for _ in range(self.N)]
        x_obsts_dyn_N = [[] for _ in range(self.N)]

        self.nr_subgoals = len(goal_struct._config)

        # --- define parameters symbolically that can be fed real-time to the controller ---#
        # goals: #

        name_dict = self.get_name_list_variables()

        for i_subgoal in range(self.nr_subgoals):
            if i_subgoal < 2:
                x_goals.append(ca.SX.sym("x_goal_" + str(i_subgoal), 3, 1))
            else:
                x_goals.append(ca.SX.sym("x_goal_" + str(i_subgoal), 1, 1))
            weight_goals.append(ca.SX.sym('weight_goal_' + str(i_subgoal), 1, 1))
        if self.nr_subgoals>1:
            angle_goals_1 = [ca.SX.sym('angle_goal_1', 3, 3)]
        else:
            angle_goals_1 = []

        # radius of the ego-robot bodies: #
        if self.nr_obsts + self.nr_obsts_dyn > 0:
            for i_coll in self.collision_links_nrs:
                if i_coll > 2:
                    radius_bodies.append(ca.SX.sym("radius_body_panda_link" + str(i_coll)))

        # static obstacles: #
        for i_obst in range(self.nr_obsts):
            radius_obsts.append(ca.SX.sym("radius_obst_" + str(i_obst)))
            x_obsts.append(ca.SX.sym("x_obst_" + str(i_obst), 3, 1))

        # dynamic obstacles: start positions: #
        for i_obst_dyn in range(self.nr_obsts_dyn):
            radius_obsts_dyn.append(ca.SX.sym("radius_obst_" + str(i_obst_dyn) + "_dynamic"))
            x_obsts_dyn.append(ca.SX.sym("x_obst_" + str(i_obst_dyn), 3, 1))
            v_obsts_dyn.append(ca.SX.sym("v_obst_" + str(i_obst_dyn), 3, 1))
            a_obsts_dyn.append(ca.SX.sym("a_obst_" + str(i_obst_dyn), 3, 1))

        # plane constraints: #
        for i_constraint in range(self.nr_constraints):
            constraints.append(ca.SX.sym("constraint_" + str(i_constraint), 4, 1))

        # state (position and velocity): #
        q_robots = ca.SX.sym("q", self.dof, 1)
        q_dot_robots = ca.SX.sym("q_dot", self.dof, 1)

        # action symbolically:
        input_keys = planner._funs._input_keys
        action_fun = planner._funs._function

        q_cur = copy.deepcopy(q_robots)
        q_dot_cur = copy.deepcopy(q_dot_robots)
        x_obsts_dyn_0 = copy.deepcopy(x_obsts_dyn)

        # loop over the control horizon:
        for k in range(self.N):
            args_action = [*angle_goals_1, *constraints,
                           q_robots, q_dot_robots,
                           *radius_bodies, *radius_obsts, *radius_obsts_dyn,
                           *weight_goals, *x_goals,
                           *x_obsts, *x_obsts_dyn, *a_obsts_dyn,
                           *v_obsts_dyn]

            if self.fabrics_mode == "vel":
                q_dot_robots = action_fun(*args_action)
                q_ddot_robots = np.zeros((self.dof))
            else:
                q_ddot_robots = action_fun(*args_action)

            if self.fabrics_mode == "acc":
                q_robots, q_dot_robots = self.system_step(pos=q_robots,
                                                          vel=q_dot_robots,
                                                          input=q_ddot_robots,
                                                          dt=self.dt,
                                                          fabrics_mode=self.fabrics_mode)
            elif self.fabrics_mode == "vel":
                q_robots, q_dot_robots= self.system_step(pos=q_robots,
                                                         vel=[],
                                                         input=q_dot_robots,
                                                         dt=self.dt,
                                                         fabrics_mode=self.fabrics_mode)

            for i_obst_dyn in range(self.nr_obsts_dyn):
                x_obsts_dyn[i_obst_dyn], _ = self.system_step(pos=x_obsts_dyn[i_obst_dyn],
                                                              vel=[],
                                                              input=v_obsts_dyn[i_obst_dyn],
                                                              dt=self.dt,
                                                              fabrics_mode="vel")

            q_robots_N[k] = copy.deepcopy(q_robots)
            q_dot_robots_N[k] = copy.deepcopy(q_dot_robots)
            q_ddot_robots_N[k] = copy.deepcopy(q_ddot_robots)
            x_obsts_dyn_N[k] = copy.deepcopy(x_obsts_dyn)

        # store q, q_dot, qddot as functions:
        args_action_fun = [*angle_goals_1,
                           *constraints,
                           q_cur, q_dot_cur,
                           *weight_goals, *x_goals,
                           *x_obsts, *radius_obsts,
                           *radius_bodies,
                           *radius_obsts_dyn,
                           *x_obsts_dyn_0, *v_obsts_dyn, *a_obsts_dyn]
        args_action_names = name_dict["angle_goals_1"]+name_dict["constraints"]+\
                            name_dict["q_robots"] + name_dict["q_dot_robots"]+\
                            name_dict["weight_goals"] + name_dict["x_goals"]+\
                            name_dict["x_obsts"] + name_dict["radius_obsts"]+\
                            name_dict["radius_bodies"] + name_dict["radius_obsts_dyn"]+\
                            name_dict["x_obsts_dyn"] + name_dict["v_obsts_dyn"] + name_dict["a_obsts_dyn"]

        self.q_N_fun = ca.Function("q_N", args_action_fun, q_robots_N, args_action_names, name_dict["q_robots_N"])
        self.q_dot_N_fun = ca.Function("q_dot_N", args_action_fun, q_dot_robots_N, args_action_names, name_dict["q_dot_robots_N"])
        self.q_ddot_N_fun= ca.Function("q_ddot_N", args_action_fun, q_ddot_robots_N, args_action_names, name_dict["q_ddot_robots_N"])

        # store dynamic obstacle positions as functions:
        args_dyn_fun = [*x_obsts_dyn_0, *v_obsts_dyn, *a_obsts_dyn]
        args_dyn_names = name_dict["x_obsts_dyn"]+name_dict["v_obsts_dyn"]+name_dict["a_obsts_dyn"]
        self.x_obsts_dyn_N_fun = [ca.Function("q_N", args_dyn_fun, x_obsts_dyn_N[k], args_dyn_names, name_dict["x_obsts_dyn_N"][k]) for k in range(self.N)]

        # get average velocity function for deadlock avoidance:
        avg_vel = self.compute_velocity_average(q_dot_robots_N)
        self.avg_vel_fun = ca.Function("avg_vel", args_action_fun, avg_vel, args_action_names, ["v_avg"])

        return {}

    def x_obsts_dyn_numerical(self, pos_obsts_dyn):
        """
        Using the symbolic function of x_obsts_dyn_fun, insert numerical values and retrieve the output:
        output: list of numerical values of the dynamic obstacles along the horizon.
        """
        args_fun = [*pos_obsts_dyn, *self.v_obsts_dyn, *self.a_obsts_dyn]
        x_obsts_dyn_N = []
        for k in range(self.N):
            solution_dm = self.x_obsts_dyn_N_fun[k](*args_fun)
            if type(solution_dm) is tuple:
                x_obsts_dyn_k = np.array([solution_i.full() for solution_i in list(solution_dm)]).transpose()[0]
            else:
                x_obsts_dyn_k = np.array([solution_i.full() for solution_i in [solution_dm]]).transpose()[0]
            x_obsts_dyn_N.append(x_obsts_dyn_k)
        return x_obsts_dyn_N

    def define_arguments_numerical(self, q_robot, q_dot_robot, weight_goals, x_goals, x_obsts, x_obsts_dyn, v_obsts_dyn, constraints=[]):
        arguments = []
        if self.nr_subgoals>1:
            arguments.append(self.rotation_matrix_panda)
        for i_constraints in range(self.nr_constraints):
            arguments.append(constraints)
        arguments.append(q_robot)
        arguments.append(q_dot_robot)
        for i_subgoal in range(self.nr_subgoals):
            arguments.append(weight_goals["subgoal" + str(i_subgoal)])
        for i_subgoal in range(self.nr_subgoals):
            arguments.append(x_goals["subgoal" + str(i_subgoal)])
        for i_obst in range(self.nr_obsts):
            arguments.append(x_obsts[i_obst])
        for i_obst in range(self.nr_obsts):
            arguments.append(self.radius_obsts[i_obst])
        if self.nr_obsts + self.nr_obsts_dyn >0:
            for r_body_robot in self.radius_body_panda_links.values():  # radius bodies
                arguments.append(r_body_robot)
        for r_other_robot in self.radius_obsts_dyn:  # radius dynamic obstacles
            arguments.append(r_other_robot)
        for i_obst_dyn in range(self.nr_obsts_dyn):
            arguments.append(x_obsts_dyn[i_obst_dyn])
        for i_obst_dyn in range(self.nr_obsts_dyn):
            arguments.append(v_obsts_dyn[i_obst_dyn])
        for i_obst_dyn in range(self.nr_obsts_dyn):
            arguments.append(self.a_obsts_dyn[i_obst_dyn])

        self.arguments = arguments
        return arguments

    def rollouts_numerical(self, arguments):
        """
        Retrieve the numerical values of the joint positions, velocities and accelerations given the symbolic functions
        given:
        - the current joint position of the robot
        - the current joint velocity of the robot
        - the planar constraints
        - the weight of the goals
        - the cartesian position of the goals
        - the cartesian positions of the static obstacles
        - the cartesian positions of the dynamic obstacles along the horizon.
        """
        solution_dm = self.q_N_fun(*arguments)
        q_num_N = np.array([solution_i.full() for solution_i in list(solution_dm)]).transpose()[0]

        solution_dm = self.q_dot_N_fun(*arguments)
        q_dot_num_N = np.array([solution_i.full() for solution_i in list(solution_dm)]).transpose()[0]

        solution_dm = self.q_ddot_N_fun(*arguments)
        q_ddot_num_N = np.array([solution_i.full() for solution_i in list(solution_dm)]).transpose()[0]

        return q_num_N, q_dot_num_N, q_ddot_num_N

    def get_velocity_rollouts(self, arguments):
        avg_vel_num = self.avg_vel_fun(*arguments)
        return avg_vel_num