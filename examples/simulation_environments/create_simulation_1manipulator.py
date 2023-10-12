import gymnasium as gym
import os
import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.dynamic_sphere_obstacle import DynamicSphereObstacle

import quaternionic
class Define_Environment(object):
    def __init__(self, radius_sphere=0.08, n_collision_links_nrs=[[7]], robot_types = ["panda"],
                 v_obsts_dyn=[np.zeros((3,)), np.zeros((3,))], pos_obsts_dyn=[np.zeros((3,)), np.zeros((3,))],
                 pos_obsts=[np.zeros((3,)), np.zeros((3,))], radius_obstacle=0.1):
        robot_type = "panda.urdf"
        self.URDF_file_panda = os.path.dirname(os.path.abspath(__file__)) + "/urdfs/panda_with_finger.urdf"
        self.collision_links_nrs = n_collision_links_nrs
        self.radius_sphere=radius_sphere
        self.robot_types = robot_types
        self.n_obst_per_link = 1
        self.rgba_obstacle = [1.0, 0.0, 0.0, 1.0]
        self.v_obsts_dyn = v_obsts_dyn
        self.pos_obsts_dyn = pos_obsts_dyn
        self.pos_obsts = pos_obsts
        self.radius_obstacle = radius_obstacle

    def initialize_environment_panda(self, render=True, nr_obst=1, nr_obsts_dyn=0, fabrics_mode="acc"):
        """
        Initializes the simulation environment.
        Adds obstacles and goal visualizaion to the environment based and
        steps the simulation once.
        """
        robots = [
            GenericUrdfReacher(urdf=self.URDF_file_panda, mode=fabrics_mode),
        ]
        env: UrdfEnv  = gym.make(
            "urdf-env-v0",
            dt=0.01, robots=robots, render=render
        )
        full_sensor = FullSensor(goal_mask=["position"], obstacle_mask=["position", "size"], variance=0)
        # Definition of the obstacle.
        obstacles = []
        for i_obst in range(nr_obst):
            static_obst_dict = {
                "type": "sphere",
                "geometry": {"position": self.pos_obsts[i_obst], "radius": self.radius_obstacle},
                "rgba": self.rgba_obstacle
            }
            obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
            obstacles.append(obst1)
        for i_obst_dyn in range(nr_obsts_dyn):
            dynamic_obst_dict = {
                "type": "sphere",
                "geometry": {"trajectory": [str(self.pos_obsts_dyn[i_obst_dyn][0])+" + t * "+str(self.v_obsts_dyn[i_obst_dyn][0]),
                                            str(self.pos_obsts_dyn[i_obst_dyn][1])+" + t * "+str(self.v_obsts_dyn[i_obst_dyn][1]),
                                            str(self.pos_obsts_dyn[i_obst_dyn][2])+" + t * "+str(self.v_obsts_dyn[i_obst_dyn][2])], "radius": self.radius_obstacle},
                "rgba": self.rgba_obstacle
            }
            obst3 = DynamicSphereObstacle(name="dynamicObst", content_dict=dynamic_obst_dict)
            obstacles.append(obst3)
        # dynamic_obst_dict = {
        #     "type": "sphere",
        #     "geometry": {"trajectory": ["-3 + t * 0.1", "0.6", "0.8"], "radius": 0.1},
        #     "rgba": self.rgba_obstacle
        # }
        # obst4 = DynamicSphereObstacle(name="dynamicObst", content_dict=dynamic_obst_dict)
        # Definition of the goal.
        goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "panda_link0",
                "child_link": "panda_link9",
                "desired_position": [0.1, -0.6, 0.4],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            # "subgoal1": {
            #     "weight": 5.0,
            #     "is_primary_goal": False,
            #     "indices": [0, 1, 2],
            #     "parent_link": "panda_link7",
            #     "child_link": "panda_hand",
            #     "desired_position": [0.1, 0.0, 0.0],
            #     "epsilon": 0.05,
            #     "type": "staticSubGoal",
            # }
        }
        goal = GoalComposition(name="goal", content_dict=goal_dict)
        # obstacles = (obst1, obst2, obst3, obst4)
        # initial_positions = np.zeros(7)
        # initial_positions[3]= -1
        # initial_positions[5] = 1  #set this one a bit further from its boundary
        env.reset()
        env.add_sensor(full_sensor, [0])
        for obst in obstacles:
            env.add_obstacle(obst)
        for sub_goal in goal.sub_goals():
            env.add_goal(sub_goal)
        env.set_spaces()
        env = self.add_collision_spheres(env)
        return (env, goal)


    def initialize_environment_panda_ring(self, render=True, nr_obst = 1, nr_obsts_dyn= 0, fabrics_mode="acc"):
        """
        Initializes the simulation environment.
        Adds obstacles and goal visualizaion to the environment based and
        steps the simulation once.
        """
        robots = [
            GenericUrdfReacher(urdf=self.URDF_file_panda, mode=fabrics_mode),
        ]
        env: UrdfEnv  = gym.make(
            "urdf-env-v0",
            dt=0.01, robots=robots, render=render
        )
        full_sensor = FullSensor(goal_mask=["position"], obstacle_mask=["position", "size"], variance=0)
        q0 = np.array([0.0, -1.0, 0.0, -1.501, 0.0, 1.8675, 0.0])
        # Definition of the obstacle.
        radius_ring = 0.3
        obstacles = []
        goal_orientation = [-0.366, 0.0, 0.0, 0.3305]
        rotation_matrix = quaternionic.array(goal_orientation).to_rotation_matrix
        whole_position = [0.1, 0.6, 0.8]
        for i in range(nr_obst):
            angle = i/nr_obst * 2.*np.pi
            origin_position = [
                0.0,
                radius_ring * np.cos(angle),
                radius_ring * np.sin(angle),
            ]
            position = np.dot(np.transpose(rotation_matrix), origin_position) + whole_position
            static_obst_dict = {
                "type": "sphere",
                "geometry": {"position": position.tolist(), "radius": self.radius_obstacle},
                "rgba": self.rgba_obstacle
            }
            obstacles.append(SphereObstacle(name="staticObst", content_dict=static_obst_dict))
        # Definition of the goal.
        for i_obst_dyn in range(nr_obsts_dyn):
            dynamic_obst_dict = {
                "type": "sphere",
                "geometry": {"trajectory": [str(self.pos_obsts_dyn[i_obst_dyn][0])+" + t * "+str(self.v_obsts_dyn[i_obst_dyn][0]),
                                            str(self.pos_obsts_dyn[i_obst_dyn][1])+" + t * "+str(self.v_obsts_dyn[i_obst_dyn][1]),
                                            str(self.pos_obsts_dyn[i_obst_dyn][2])+" + t * "+str(self.v_obsts_dyn[i_obst_dyn][2])], "radius": self.radius_obstacle},
                "rgba": self.rgba_obstacle
            }
            obst3 = DynamicSphereObstacle(name="dynamicObst", content_dict=dynamic_obst_dict)
            obstacles.append(obst3)
        # Definition of the goal.
        goal_orientation = [-0.366, 0.0, 0.0, 0.3305]
        rotation_matrix = quaternionic.array(goal_orientation).to_rotation_matrix
        goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "panda_link0",
                "child_link": "panda_link9",
                "desired_position": whole_position,
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            # "subgoal1": {
            #     "weight": 5.0,
            #     "is_primary_goal": False,
            #     "indices": [0, 1, 2],
            #     "parent_link": "panda_link7",
            #     "child_link": "panda_hand",
            #     "desired_position": [0.1, 0.0, 0.0],
            #     "epsilon": 0.05,
            #     "type": "staticSubGoal",
            # }
        }
        goal = GoalComposition(name="goal", content_dict=goal_dict)
        env.reset(pos=q0)
        env.add_sensor(full_sensor, [0])
        for obst in obstacles:
            env.add_obstacle(obst)
        for sub_goal in goal.sub_goals():
            env.add_goal(sub_goal)
        env.set_spaces()
        env = self.add_collision_spheres(env)
        return (env, goal)

    def add_collision_spheres(self, env):
        """
        Add the collision spheres around the manipulators.
        The number of collision spheres per link are given by n_obst_per_link which is, amongst other parameters,
        defined in define_parameters()
        """

        # --- define parameters ---#
        nr_robots = len(self.collision_links_nrs)
        dicts_collision_links = [[] for _ in range(nr_robots)]
        collision_links_urdf = [[] for _ in range(nr_robots)]
        for i_robot in range(nr_robots):
            dicts_collision_links[i_robot] = {
                'link_nr': ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6",
                            "panda_joint7", "panda_joint8"],
                'radi': [self.radius_sphere] * 8,
                'length': [0.333, 0.2, 0.3164, 0.2, 0.3840, 0.2, 0.088, 0.2],
                'type': ['linear', 'rotational', 'linear', 'rotational', 'linear', 'rotational', 'linear', 'rotational'],
                'axis': ['z', 'z', 'z', 'z', 'z', 'z', 'z', 'z', 'z']}

        # --- For each robot define the collision links ---#
        for i_robot in range(nr_robots):
            links_urdf = self.collision_links_nrs[i_robot]
            if self.robot_types[i_robot] == "panda":
                joint_map_urdf_env = env.env.env._robots[0]._urdf_robot._joint_map
                links_urdf = [list(joint_map_urdf_env.keys()).index(link) for link in
                              dicts_collision_links[i_robot]["link_nr"]]
            dicts_collision_links[i_robot]["link_nr_urdf"] = links_urdf

            for col_link in self.collision_links_nrs[i_robot]:
                collision_links_urdf[i_robot].append(dicts_collision_links[i_robot]["link_nr_urdf"][col_link - 1])

            # --- find the correct z-transform for each sphere, and to which link the sphere belongs ---#
            for i_link, link in enumerate(collision_links_urdf[i_robot]):
                for i in range(self.n_obst_per_link):
                    dict_collision_links = dicts_collision_links[i_robot]
                    link_transformation = np.identity(4)
                    idx = dict_collision_links['link_nr_urdf'].index(link)
                    if dict_collision_links['type'][idx] == 'linear':
                        if i_robot == 1 and link == 4:
                            z_start = dict_collision_links['length'][idx]
                        else:
                            z_start = dict_collision_links['length'][idx]
                    else:
                        z_start = dict_collision_links['length'][idx] / 2
                    z_range = dict_collision_links['length'][idx]
                    z_transform = -z_start + i * z_range / self.n_obst_per_link
                    if dict_collision_links['axis'][idx] == 'z':
                        link_transformation[0:3, 3] = [0, 0, z_transform]
                    elif dict_collision_links['axis'][idx] == 'y':
                        link_transformation[0:3, 3] = [0, z_transform, 0]
                    else:
                        link_transformation[0:3, 3] = [z_transform, 0, 0]
                    # since the gripper is quite big, I am defining two spheres that are translated also on x and y to capture the geometry:
                    if link == 16:
                        x_transform = 0.03
                        y_transform = 0.03
                        if i == 1:
                            z_transform = -z_start + (i + 1) * z_range / self.n_obst_per_link
                            link_transformation[0:3, 3] = [x_transform, y_transform, z_transform]
                        elif i == 2:
                            link_transformation[0:2, 3] = [-x_transform, -y_transform]
                    # since there is a weird bend in the black link of the panda, I also translate this sphere:
                    if link == 11 and (i == 2 or i == 3):
                        if i == 2:
                            y_transform = -0.02
                        else:
                            y_transform = -0.06
                        link_transformation[0:2, 3] = [0, -y_transform]

                    # --- add the collision sphere in the environment --- #
                    env.add_collision_link(
                        robot_index=i_robot,
                        link_index=link,
                        shape_type="sphere",
                        sphere_on_link_index=i,
                        link_transformation=link_transformation,
                        size=[dict_collision_links['radi'][dict_collision_links['link_nr_urdf'].index(link)]]
                    )
        return env
