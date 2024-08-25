import copy
import os
import numpy as np
import random
import gymnasium as gym
import pybullet
from typing import List
from mpscenes.obstacles.collision_obstacle import CollisionObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.goals.goal_composition import GoalComposition

class create_manipulators_simulation:

    def __init__(self, params):
        self.urdf_files = params.urdf_links
        self.z_table = params.mount_param["z_table"]
        self.mount_positions = params.mount_param["mount_positions"]
        self.mount_orientations = params.mount_param["mount_orientations"]
        self.collision_links = params.collision_links
        self.collision_links_nrs = params.collision_links_nrs
        self.n_obst_per_link = params.n_obst_per_link
        self.radius_sphere = params.radius_sphere
        self.robot_types = params.robot_types
        self.dt=params.dt
        self.pos0 = params.pos0
        self.tray_positions = params.tray_positions
        self.tray_orientations = params.tray_orientations
        self.table_position = params.table_position

        self.nr_robots = len(self.robot_types)
        self.link_transform_list = [[] for _ in range(self.nr_robots)]
        self.y_trans = 0.0
        if self.nr_robots == 3:
            self.y_trans = 0.2
        self.block_xyz_fixed = {}
        if self.nr_robots == 2:
            self.block_xyz_fixed["0"] = [
                [0.4, -0.0+self.y_trans, self.z_table + 0.07],
                [0.4, -0.15+self.y_trans, self.z_table + 0.07],
                [0.4, 0.15+self.y_trans, self.z_table + 0.07],
            ]
            self.block_xyz_fixed["1"] = [
                [0.6, -0.15+self.y_trans, self.z_table + 0.07],
                [0.6, 0.0+self.y_trans, self.z_table + 0.07],
                [0.6, 0.15+self.y_trans, self.z_table + 0.07],
            ]
        elif self.nr_robots == 3:
            self.block_xyz_fixed["0"] = [
                [0.4, -0.0+self.y_trans, self.z_table + 0.07],
                [0.4, -0.15+self.y_trans, self.z_table + 0.07],
            ]
            self.block_xyz_fixed["1"] = [
                [0.6, -0.15+self.y_trans, self.z_table + 0.07],
                [0.6, 0.0+self.y_trans, self.z_table + 0.07],
                [0.6, 0.15+self.y_trans, self.z_table + 0.07],
            ]
            self.block_xyz_fixed["2"] = [
                [0.4, 0.15+self.y_trans, self.z_table + 0.07],
                [0.6, 0.15+self.y_trans, self.z_table + 0.07],
            ]

    def check_cube_validity(self, new_cube: BoxObstacle, existing_cubes: list) -> bool:
        """
        Check if the cube is placed within reasonable distance from another cube.
        To avoid cubes that are placed on top of each other or too close to grasp.

        Input: Struct of the new cube
        Output: If the cube is valid or not (bool).
        """
        slack = 0.06  # allowed distance between blocks [m]
        pos_new = new_cube._config.geometry.position
        for existing_cube in existing_cubes:
            diff = np.linalg.norm(np.array(pos_new) - np.array(existing_cube._config.geometry.position))
            if diff <= max([new_cube._config.geometry.length, new_cube._config.geometry.width]) / 2 + max(
                    [existing_cube._config.geometry.length, existing_cube._config.geometry.width]) / 2 + slack:
                return False
        return True

    def create_scene(self, random_scene=False, n_cubes=6) -> List[CollisionObstacle]:
        """
        Create a random or a fixed scene with n_cubes the number of cubes.
        Input: If the scene is random or fixed, and the number of cubes
        Output: A list of collision obstacles of a specific struct
        """
        obstacles = []

        # --- random scene ---#
        if random_scene:
            while len(obstacles) < n_cubes:
                if len(obstacles) < n_cubes / self.nr_robots:
                    rgba_cube = [0.0, 1.0, 0.0, 1.0]
                elif len(obstacles) < n_cubes / self.nr_robots * 2:
                    rgba_cube = [1.0, 0.0, 0.0, 1.0]
                else:
                    rgba_cube = [0.0, 0.0, 1.0, 1.0]
                object_dict = {
                    "type": "box",
                    "movable": True,
                    "geometry": {
                        "position": [random.uniform(0.4, 0.6), random.uniform(-0.15, 0.15)+self.y_trans, self.z_table + 0.07],
                        "orientation": [0, 0, 0, 1],
                        "length": 0.05,
                        "height": 0.05,
                        "width": 0.05,
                    },
                    "rgba": rgba_cube
                }
                object_new = BoxObstacle(name="cube{}".format(len(obstacles) - 2), content_dict=object_dict)
                if self.check_cube_validity(object_new, obstacles):
                    obstacles.append(object_new)

        # --- Fixed scene with blocks --- #
        else:
            for i_robot in range(self.nr_robots):
                for i in range(int(n_cubes/self.nr_robots)):
                    if len(obstacles) < n_cubes / self.nr_robots:
                        rgba_cube = [0.0, 1.0, 0.0, 1.0]
                    elif len(obstacles) < n_cubes / self.nr_robots * 2:
                        rgba_cube = [1.0, 0.0, 0.0, 1.0]
                    else:
                        rgba_cube = [0.0, 0.0, 1.0, 1.0]
                    object_dict = {
                        "type": "box",
                        "movable": True,
                        "geometry": {
                            "position": [self.block_xyz_fixed[str(i_robot)][i][0], self.block_xyz_fixed[str(i_robot)][i][1], self.z_table + 0.07],
                            "orientation": [0, 0, 0, 1],
                            "length": 0.05,
                            "height": 0.05,
                            "width": 0.05,
                        },
                        "rgba": rgba_cube
                    }
                    object_block = BoxObstacle(name="cube1", content_dict=object_dict)
                    obstacles.append(object_block)
        return obstacles
    #
    def initialize_environment(self, render=True, random_scene=False, obstacles=[]):
        """
        Initializes the simulation environment.

        Adds obstacles and goal visualizaion to the environment based and
        steps the simulation once.
        """
        robots = []
        for i_robot in range(self.nr_robots):
            robots.append(GenericUrdfReacher(urdf=self.urdf_files["URDF_file_panda"], mode="vel"))
        env: UrdfEnv = UrdfEnv(
            robots=robots,
            dt=self.dt,
            render=render,
            observation_checking=False,
        )
        self.fk_robots = robots
        full_sensor = FullSensor(
            goal_mask=[],
            obstacle_mask=["position", "size"],
            variance=0,
        )
        env.reset(pos=self.pos0, mount_positions=self.mount_positions, mount_orientations=self.mount_orientations)
        # for i in range(9, 13): # if you want, the dynamics of the pybullet simulator can be changed for the joints
        #     pybullet.changeDynamics(2, i, lateralFriction=200) #, spinningFriction=200)
        env.add_sensor(full_sensor, [0])
        trayUid_panda1 = pybullet.loadURDF(self.urdf_files["URDF_tray_location"], basePosition=self.tray_positions[0],
                                           baseOrientation=self.tray_orientations[0])
        trayUid_panda2 = pybullet.loadURDF(self.urdf_files["URDF_tray_location"], basePosition=self.tray_positions[1],
                                           baseOrientation=self.tray_orientations[1])
        tableUid = pybullet.loadURDF(self.urdf_files["URDF_table"], basePosition=self.table_position)
        for obst in obstacles:
            env.add_obstacle(obst)
        env.set_spaces()
        env = self.add_collision_spheres(env)
        return env

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
                joint_map_urdf_env = env._robots[0]._urdf_robot._joint_map
                links_urdf = [list(joint_map_urdf_env.keys()).index(link) for link in
                              dicts_collision_links[i_robot]["link_nr"]]
            dicts_collision_links[i_robot]["link_nr_urdf"] = links_urdf

            for col_link in self.collision_links_nrs[i_robot]:
                collision_links_urdf[i_robot].append(dicts_collision_links[i_robot]["link_nr_urdf"][col_link - 1])

            # --- find the correct z-transform for each sphere, and to which link the sphere belongs ---#
            for i_link, link in enumerate(collision_links_urdf[i_robot]):
                link_transforms = []
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
                    link_transforms.append(link_transformation)
                self.link_transform_list[i_robot].append(link_transforms)
        return env

    def get_link_transforms(self):
        return self.link_transform_list
