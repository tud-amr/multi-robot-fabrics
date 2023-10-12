import os
import numpy as np
import copy
class manipulator_parameters():

    def __init__(self, nr_robots,n_obst_per_link):
        # --- parameters ---- #
        self.dt = 0.01  # sample time
        self.n_cubes = 6  # number of cubes
        self.nr_robots = nr_robots
        self.dof = [7]*nr_robots  # degrees of freedom per robot
        self.fabrics_mode = "vel"
        self.nu = self.dof
        if self.fabrics_mode == "acc":
            self.nx = self.dof * 2
        elif self.fabrics_mode == "vel":
            self.nx = self.dof
        else:
            self.nx = self.dof * 2
        self.nr_obsts = [0]*nr_robots  # number of static obstacles considered per robot
        self.radius_obsts = [[]*nr_robots]
        self.n_obst_per_link = n_obst_per_link  # number of obstacles per link on the robots
        self.radius_sphere = 0.08  # radius of the collision spheres
        self.nr_constraints = [1]*self.nr_robots
        self.collision_links_nrs = [[1, 2, 3, 4, 5, 6, 7, 8]]*nr_robots
        self.collision_links = [["panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7","panda_link8"]]*nr_robots

        self.nr_obsts_dyn = [0 for _ in range(nr_robots)]
        self.nr_obsts_dyn_all = [0 for _ in range(nr_robots)]
        for i_robot in range(nr_robots):
            nr_obsts_i = sum([len(self.collision_links_nrs[i]) for i in range(nr_robots) if i != i_robot])
            self.nr_obsts_dyn[i_robot] = self.nr_obsts_dyn[i_robot] + nr_obsts_i
            self.nr_obsts_dyn_all[i_robot] = self.nr_obsts_dyn_all[i_robot] + nr_obsts_i * self.n_obst_per_link
        self.robot_types = ["panda"]*nr_robots

        self.r_robots = copy.deepcopy(self.collision_links_nrs)
        self.radius_body_panda_links = {}
        for i_robot in range(len(self.collision_links_nrs)):
            for j, col_link in enumerate(self.collision_links_nrs[i_robot]):
                self.r_robots[i_robot][j] = self.radius_sphere
                if self.robot_types[i_robot] == "panda" and col_link > 2:
                    # radii on ego body are not considered for obstacle avoidance until link 2, since they are fixed
                    self.radius_body_panda_links[str(col_link)] = np.array(self.radius_sphere)

        self.a_dyns_obsts = [[] for _ in range(nr_robots)]
        self.r_dyns_obsts = [[] for _ in range(nr_robots)]
        for i_robot in range(nr_robots):
            i_other_robots = [i for i in range(nr_robots) if i != i_robot]
            for i_other_robot in i_other_robots:
                for i_sphere in range(self.n_obst_per_link*len(self.collision_links_nrs[i_other_robot])):
                    self.a_dyns_obsts[i_robot] = self.a_dyns_obsts[i_robot] + [np.zeros((3,))] * self.n_obst_per_link  # [a_robots[i_other_robot][i_sphere]] * n_obst_per_link
                    self.r_dyns_obsts[i_robot] = self.r_dyns_obsts[i_robot] + [self.r_robots[i_other_robot][i_sphere]] * self.n_obst_per_link

        # --- initialize settings --- #
        self.ROLLOUT_FABRICS = False
        self.ROLLOUTS_PLOTTING = False
        self.STATIC_OR_DYN_FABRICS = 0
        self.RESOLVE_DEADLOCKS = True
        self.ESTIMATE_GOAL = False
        self.N_HORIZON = 2
        self.MPC_LAYER = False

        # --- URDF file locations --- #
        self.URDF_file_panda = os.path.dirname(os.path.abspath(__file__)) + "/simulation_environments/urdfs/panda_with_finger.urdf"
        self.URDF_file_kinova = os.path.dirname(os.path.abspath(__file__)) + "/simulation_environments/urdfs/kinova_gen_3_lite.urdf"
        if nr_robots == 2:
            self.URDF_tray_location = os.path.dirname(os.path.abspath(__file__)) + "/simulation_environments/urdfs/tray/tray.urdf"
            self.URDF_table = os.path.dirname(os.path.abspath(__file__)) + "/simulation_environments/urdfs/table/table.urdf"
        elif nr_robots == 3:
            self.URDF_tray_location = os.path.dirname(os.path.abspath(__file__)) + "/simulation_environments/urdfs/tray/tray_shallow.urdf"
            self.URDF_table = os.path.dirname(os.path.abspath(__file__)) + "/simulation_environments/urdfs/table/table_enlarged.urdf"
        else:
            print("error, nr of robots is not possible, change to 2 or 3")

        self.urdf_links = {"URDF_file_panda": self.URDF_file_panda,
                      "URDF_file_kinova": self.URDF_file_kinova,
                      "URDF_tray_location": self.URDF_tray_location,
                      "URDF_table": self.URDF_table}

        # --- global position and orientation in the room -- #
        self.z_table = 0.65

        if nr_robots == 2:
            self.mount_positions = [
                np.array([0.0, 0.0, self.z_table]),
                np.array([1.0, 0.0, self.z_table]),
            ]
            self.mount_orientations = [
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, 1.0, 0.0]),
            ]
            self.pos0 = np.array([
                np.array([1.125, 0.19, 0.12, -1.66, -0.0, 1.88, np.pi / 4, 0.02, 0.02]),
                np.array([1.125, 0.19, 0.12, -1.66, -0.0, 1.88, np.pi / 4, 0.02, 0.02])
            ])
            self.tray_positions = [[0.2, 0.67, 0.57], [0.9, -0.67, 0.57]]
            self.tray_orientations = [[0, 0, -1, 1], [0, 0, 1, 1]]
            self.table_position = [0.5, 0, 0]

        elif nr_robots == 3:
            self.mount_positions = [
                np.array([0.0, 0.0, self.z_table]),
                np.array([1.0, 0.0, self.z_table]),
                np.array([0.7, 0.6, self.z_table])
            ]
            self.mount_orientations = [
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0, 0.0])
            ]
            self.pos0 =  np.array([
                        np.array([1.13793529, -0.3227085, -0.02767777, -2.2204281, -0.00917029, 1.88612235, 0.78536134]),
                        np.array([1.131,  0.20,  0.12, -1.65, -0.0, 1.86,  np.pi / 4, 0.02, 0.02]),
                        np.array([-0.46609715, -0.25025564, -0.40425878, -2.0966941 , -0.10682593, 1.84917516,  0.37170524])
                ])
            self.tray_positions = [[0.2, 0.77, 0.64], [0.9, -0.63, 0.62]]
            self.tray_orientations = [[0, 0, 0, 1], [0, 0, 1, 1]]
            self.table_position = [0.5, 0.4, 0]
        else:
            print("error, nr of robots is not possible, change to 2 or 3")
        rotation_matrix_panda = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.rotation_matrix_pandas = [rotation_matrix_panda] * nr_robots
        self.mount_param = {"z_table": self.z_table, "mount_positions": self.mount_positions, "mount_orientations": self.mount_orientations}

        # --- Start xyz position of the robots --- #
        if self.nr_robots == 2:
            self.start_goals = [[0.2, 0.6, self.mount_param["z_table"] + 0.5],
                                [0.8, -0.6, self.mount_param["z_table"] + 0.5]]
        elif self.nr_robots == 3:
            self.start_goals = [[0.25, 0.6, self.z_table + 0.5],
                                [0.8, -0.5, self.z_table + 0.5],
                                [0.4, 0.5, self.z_table + 0.3]]
        else:
            print("error, nr of robots is not possible, change to 2 or 3")

        self.constraints = [np.array([0, 0, 1, 0.0 - self.mount_param["z_table"]])] * self.nr_robots

        self.mount_transform = [[] for _ in range(nr_robots)]
        for i_robot in range(nr_robots):
            if i_robot == 1:
                angle_rot = np.pi
            elif i_robot == 2:
                angle_rot = np.pi
            else:
                angle_rot = 0.0
            T_0 = np.identity(4)
            rot_ang_r1 = angle_rot  # np.pi
            T_0[0:2, 0:2] = np.array([[np.cos(rot_ang_r1), -np.sin(rot_ang_r1)], [np.sin(rot_ang_r1), np.cos(rot_ang_r1)]])
            T_0[0:3, 3] = self.mount_param["mount_positions"][i_robot]
            self.mount_transform[i_robot] = T_0
    def get_mount_parameters(self) -> dict:
        return self.mount_param

    def get_urdf_locations(self) -> dict:
        return self.urdf_links

    def define_settings(self, ROLLOUT_FABRICS=False, ROLLOUTS_PLOTTING=False, STATIC_OR_DYN_FABRICS=0, RESOLVE_DEADLOCKS=True,
                        ESTIMATE_GOAL=False, N_HORIZON=10, MPC_LAYER=False, n_obst_per_link=1) -> list:
        """
        Define some variables:
        - ROLLOUT_FABRICS: if forward simulations are performed
        - ROLLOUT_PLOTTING: if the rollouts are plotted and stored along the horizon (state, input) or only the average velocity
        - STATIC_OR_DYN_FABRICS: 0 if static (zero velocity obstacles) or 1 if dynamic fabrics (constant velocity obstacles) is used
        - RESOLVE_DEADLOCKS: if deadlocks are resolved
        - ESTIMATE_GOAL: if the goal is communicated or estimated
        - N_HORIZON: the length of the prediction horizon
        """
        self.ROLLOUT_FABRICS = ROLLOUT_FABRICS
        self.ROLLOUTS_PLOTTING = ROLLOUTS_PLOTTING
        self.STATIC_OR_DYN_FABRICS = STATIC_OR_DYN_FABRICS
        self.RESOLVE_DEADLOCKS = RESOLVE_DEADLOCKS
        self.ESTIMATE_GOAL = ESTIMATE_GOAL
        self.N_HORIZON = N_HORIZON
        self.MPC_LAYER = MPC_LAYER
        self.n_obst_per_link = n_obst_per_link
        return [self.ROLLOUT_FABRICS, self.ROLLOUTS_PLOTTING, self.STATIC_OR_DYN_FABRICS, self.RESOLVE_DEADLOCKS, self.ESTIMATE_GOAL, self.N_HORIZON, self.MPC_LAYER, self.n_obst_per_link]

    def get_settings(self):
        return [self.ROLLOUT_FABRICS, self.ROLLOUTS_PLOTTING, self.STATIC_OR_DYN_FABRICS, self.RESOLVE_DEADLOCKS, self.ESTIMATE_GOAL, self.N_HORIZON, self.MPC_LAYER]

    def set_horizon(self, n_horizon):
        self.N_HORIZON = n_horizon