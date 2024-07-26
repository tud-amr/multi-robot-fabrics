from mpscenes.goals.goal_composition import GoalComposition
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from multi_robot_fabrics.fabrics_planner.forward_planner_Jointspace import ForwardFabricsPlanner

class CreatingPlanner:
    def __init__(self):
        self.goal_orientation = [-0.366, 0.0, 0.0, 0.3305]
        self.whole_position = [0.1, 0.6, 0.8]
        self.panda_limits = [
                [-2.8973, 2.8973],
                [-1.7628, 1.7628],
                [-2.8973, 2.8973],
                [-3.0718, -0.0698],
                [-2.8973, 2.8973],
                [-0.0175, 3.7525],
                [-2.8973, 2.8973]
            ]

    def create_dummy_goal_panda(self) -> GoalComposition:
        """
        Create a dummy goal where the numberical values of the dictionary can later be changed.
        """
        goal_dict = {
            "subgoal0": {
                "weight": 2.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "world",
                "child_link": "panda_hand",
                "desired_position": self.whole_position,
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "weight": 10.0,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": "panda_link7",
                "child_link": "panda_hand",
                "desired_position": [0.107, 0.0, 0.0],
                "angle": self.goal_orientation,
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

    def set_planner_panda(self, degrees_of_freedom: int = 7, nr_obst = 0, nr_obst_dyn=1, collision_links_nr= [5], urdf_links = {}, mount_param = {}, i_robot=0):
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
        # urdf_files, mount_param = get_global_parameters()
        with open(urdf_links["URDF_file_panda"], 'r') as file:
            urdf = file.read()
        goal = self.create_dummy_goal_panda()
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

        # --- mount position/orientation: rotation angle and corresponding transformation matrix ---#
        if i_robot == 1:
            angle_rot = np.pi
        elif i_robot == 2:
            angle_rot = np.pi
        else:
            angle_rot = 0.0
        T_0 = np.identity(4)
        rot_ang_r1 = angle_rot  #np.pi
        T_0[0:2, 0:2] = np.array([[np.cos(rot_ang_r1), -np.sin(rot_ang_r1)], [np.sin(rot_ang_r1), np.cos(rot_ang_r1)]])
        T_0[0:3, 3] = mount_param["mount_positions"][i_robot]
        planner._forward_kinematics.set_mount_transformation(T_0)

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
            limits=self.panda_limits,
        )
        planner.concretize(mode='vel', time_step=0.01)  # be careful that the inputs are here VELOCITIES!!!
        return planner, goal

    def define_planners(self, params):
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
            planner_panda_i, goal_struct_panda_i = self.set_planner_panda(degrees_of_freedom=params.dof[i_robot],
                                                                    nr_obst=nr_obst_planners[i_robot],
                                                                    nr_obst_dyn=nr_obst_dyn_planners[i_robot],
                                                                    collision_links_nr=params.collision_links_nrs[i_robot],
                                                                    urdf_links = params.urdf_links,
                                                                    mount_param = params.mount_param,
                                                                    i_robot=i_robot)
            planner_panda_grasp_i, _= self.set_planner_panda(degrees_of_freedom=params.dof[i_robot],
                                                                             nr_obst=i_robot,
                                                                             nr_obst_dyn=i_robot,
                                                                             collision_links_nr=[],
                                                                             urdf_links=params.urdf_links,
                                                                             mount_param=params.mount_param,
                                                                             i_robot=i_robot)
            planners.append(planner_panda_i)
            goal_structs.append(goal_struct_panda_i)
            planners_grasp.append(planner_panda_grasp_i)
        return planners, planners_grasp, goal_structs

    def define_rollout_planners(self, params, fk_dict=None, goal_structs=None, n_steps=100):
        """
        Define the planners that are used for the rollouts (less obstacles per link)
        """
        planners_rollout = []
        for i_robot in range(params.nr_robots):
            planner_panda_fk_i, _ = self.set_planner_panda(degrees_of_freedom=params.dof[i_robot],
                                                                   nr_obst=params.nr_obsts[i_robot],
                                                                   nr_obst_dyn=params.nr_obsts_dyn[i_robot],
                                                                   collision_links_nr=params.collision_links_nrs[i_robot],
                                                                   urdf_links=params.urdf_links,
                                                                   mount_param=params.mount_param,
                                                                   i_robot=i_robot)
            planners_rollout.append(planner_panda_fk_i)

        forwardplanner = ForwardFabricsPlanner(params=params,
                                               planners=planners_rollout,
                                               N_steps=n_steps,
                                               fk_dict=fk_dict,
                                               goal_struct_robots=goal_structs)
        forwardplanner.forward_multi_fabrics_symbolic()
        return forwardplanner
