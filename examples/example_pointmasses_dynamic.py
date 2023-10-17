import gymnasium as gym
import os
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
import logging
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

logging.basicConfig(level=logging.INFO)
"""
Fabrics example with four point-mass robots and four static obstacles where the robots try to cross 
The robot see the static obstacles as static and the other robots as dynamic obstacles (with constant velocity).
"""


def initialize_environment(render, n_static_obstacles, n_robots, obstacles_pos, obstacles_radius, robots_pos, robots_goal):
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.

    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    assert len(obstacles_pos) == n_static_obstacles, "number of static obstacles has to match number of provided obstacle positions"
    assert len(obstacles_radius) == n_static_obstacles, "number of static obstacles has to match number of provided obstacle radii"

    robots = [
        GenericUrdfReacher(urdf=dir_path + "/simulation_environments/urdfs/pointRobot1.urdf", mode="acc"),
        GenericUrdfReacher(urdf=dir_path +"/simulation_environments/urdfs/pointRobot.urdf", mode="acc"),
        GenericUrdfReacher(urdf=dir_path + "/simulation_environments/urdfs/pointRobot1.urdf", mode="acc"),
        GenericUrdfReacher(urdf=dir_path +"/simulation_environments/urdfs/pointRobot.urdf", mode="acc"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # Set the initial position and velocity of the point mass.
    full_sensor = FullSensor(goal_mask=["position"], obstacle_mask=["position", "size"])
    # Definition of the obstacle.

    obstacles = []
    for i in range(n_static_obstacles):
        static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": obstacles_pos[i], "radius": obstacles_radius[i]},
        }
        obst = SphereObstacle(name="staticObst{}".format(i+1), content_dict=static_obst_dict)
        obstacles.append(obst)

    # Definition of the goal.
    goal = list(robots_goal[0])
    goal_dict = {
        "subgoal0": {
            "weight": 1,
            "is_primary_goal": True,
            "indices": [0, 1],
            "parent_link": 'world',
            "child_link": 'base_link',
            "desired_position": [1.5, 0.99],
            "epsilon": 0.1,
            "type": "staticSubGoal"
        }
        # "subgoal1": {
        #     "weight": 1,
        #     "is_primary_goal": False,
        #     "indices": [0, 1],
        #     "parent_link": 0,
        #     "child_link": 1,
        #     "desired_position": [0, 0.5],
        #     "epsilon": 0.1,
        #     "type": "staticSubGoal"
        # }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    # n = env.n()
    ns_per_robot = env.ns_per_robot()
    robots_vel = np.array([np.zeros(n) for n in ns_per_robot])
    # n_per_robot = env.n_per_robot()

    env.reset(pos=robots_pos, vel=robots_vel)

    # env.reset(pos=pos0, vel=vel0)
    for i in range(len(robots)):
        env.add_sensor(full_sensor, [i])

    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    return (env, goal)

def set_planner_point(goal: GoalComposition, n_obstacles: int = 2, n_dyn_obstacles=0):
    degrees_of_freedom = 3
    robot_type = "pointRobot"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    urdf_file = dir_path + "/simulation_environments/urdfs/pointRobot1.urdf"
    with open(urdf_file, 'r') as file:
        urdf = file.read()
    fk = GenericURDFFk(urdf, 'world', 'base_link')
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    planner = ParameterizedFabricPlanner(
            degrees_of_freedom,
            fk,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler
    )
    collision_links = ['base_link']
    self_collision_links = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links,
        self_collision_links,
        goal=goal,
        number_obstacles=n_obstacles,
        number_dynamic_obstacles=n_dyn_obstacles,
        dynamic_obstacle_dimension =2,
    )
    planner.concretize()
    return planner

def run_point_example(n_steps=1000, render=True):
    """
    Set the gym environment, the planner and run point robot example.

    Params
    ----------
    n_steps
        Total number of simulation steps.
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    n_static_obstacles = 6
    n_robots = 4

    obstacles_pos = [[1, 1.25, 0], [1, 3.75, 0], [1, -1.25, 0], [-1.1, 0, 0], [-1.1, 2.5, 0] ,[-1.1, -2.5, 0]]
    obstacles_radius =  [1, 1, 1, 1, 1,1]
    robots_pos = np.array([[-2.5, 0.01, 0.0], [-2.5, -2.49 , 0.0], [2.5, 1.26, 0.0], [2.5, 3.76, 0.0]])
    goal_robots = [np.array([1.5,3.76]), np.array([1.5, 1.26]), np.array([-2.5, 0.01]), np.array([-2.5, -2.49])]
    robots_pos = np.array([[-2.5, 0.01, 0.0], [-2.5, -2.49 , 0.0], [2.5, 1.26, 0.0], [2.5, 3.74, 0.0]])
    goal_robots = [np.array([1.5,3.76]), np.array([1.5, 1.26]), np.array([-2.5, 0.01]), np.array([-2.5, -2.49])]
    r_robots = [np.array(0.2), np.array(0.2), np.array(0.2), np.array(0.2)]

    (env, goal) = initialize_environment(render,n_static_obstacles, n_robots, obstacles_pos, obstacles_radius, robots_pos, goal_robots)
    planner_point = set_planner_point(goal, n_obstacles=n_static_obstacles, n_dyn_obstacles=n_robots-1)
    action = np.zeros(env.n())
    ob, *_ = env.step(action)

    env.reconfigure_camera(5.59999942779541, 0, -89, (0.0, 0.0, 0.0))
    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob['robot_0']
        ob_robot1 = ob['robot_1']
        ob_robot2 = ob['robot_2']
        ob_robot3 = ob['robot_3']

        #for now: use end-effector position of panda as obstacle in other panda
        pos_robot0 = ob_robot["joint_state"]["position"][0:2]
        pos_robot1 = ob_robot1["joint_state"]["position"][0:2]
        pos_robot2 = ob_robot2["joint_state"]["position"][0:2]
        pos_robot3 = ob_robot3["joint_state"]["position"][0:2]


        robots = ['robot_0', 'robot_1', 'robot_2', 'robot_3']
        pos_robots = []
        vel_robots = []
        for robot in robots:
            pos_robots.append(ob[robot]["joint_state"]["position"][0:2])
            vel_robots.append(ob[robot]["joint_state"]["velocity"][0:2])

        for i, robot in enumerate(robots):
            pos_obs = [obst['position'][0:3] for obst in list(ob[robot]['FullSensor']['obstacles'].values())]
            pos_other_robots = [pos for j,pos in enumerate(pos_robots) if j!=i]
            vel_other_robots = [vel for j,vel in enumerate(vel_robots) if j!=i]
            radius_obs = [obst['size'] for obst in list(ob[robot]['FullSensor']['obstacles'].values())]
            radius_other_robots = [r for j,r in enumerate(r_robots) if j!=i]

            idx_start = 3*i
            idx_end = 3*i+3

            action[idx_start:idx_end] = planner_point.compute_action(
                q=ob[robot]["joint_state"]["position"][0:3],
                qdot=ob[robot]["joint_state"]["velocity"][0:3],
                x_goal_0=goal_robots[i],
                weight_goal_0=goal.sub_goals()[0].weight(),
                x_obsts=pos_obs,
                radius_obsts=radius_obs,
                radius_body_base_link=r_robots[i],
                x_obst_dynamic_0=pos_other_robots[0],  #this is hardcoded (ugly), but syntax with combined dynamic obstacles not available (reported issue)
                xdot_obst_dynamic_0=vel_other_robots[0],
                xddot_obst_dynamic_0=np.array([0, 0]),  #acceleration set to zero! No dependence on fabrics of others
                radius_obst_dynamic_0=radius_other_robots[0],
                x_obst_dynamic_1=pos_other_robots[1],
                xdot_obst_dynamic_1=vel_other_robots[1],
                xddot_obst_dynamic_1=np.array([0, 0]),
                radius_obst_dynamic_1=radius_other_robots[1],
                x_obst_dynamic_2=pos_other_robots[2],
                xdot_obst_dynamic_2=vel_other_robots[2],
                xddot_obst_dynamic_2=np.array([0, 0]),
                radius_obst_dynamic_2=radius_other_robots[2],
            )
        ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_point_example(n_steps=10000, render=True)
