import os
import gymnasium as gym
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
import copy
import yaml
ROBOTTYPE = 'dingo_kinova'
ROBOTMODEL = 'dingo_kinova'

CONFIG_FILE = "configs/"+ROBOTTYPE+"_config.yaml"
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.safe_load(config_file)
    CONFIG_PROBLEM = config['problem']
    CONFIG_FABRICS = config['fabrics']

def initalize_environment(render=True, nr_obst: int = 0):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robot_model = RobotModel(ROBOTTYPE, model_name=ROBOTMODEL)
    urdf_file = robot_model.get_urdf_path()
    robots = [
        GenericUrdfReacher(urdf=urdf_file, mode="acc"),
    ]
    env: UrdfEnv = UrdfEnv(
        robots=robots,
        dt=0.01,
        render=render,
        observation_checking=False,
    )
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=['position', 'size'],
            variance=0.0
    )
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.3, -0.3, 0.3], "radius": 0.1},
    }
    obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [-0.7, 0.0, 0.5], "radius": 0.1},
    }
    obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "base_link",
            "child_link": "arm_end_effector_link",
            "desired_position": [-0.24355761, -0.75252747, 0.5],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    obstacles = [obst1, obst2][0:nr_obst]
    pos0 = np.zeros((9,))
    env.reset(pos=pos0)
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    collision_keys_robot0 = [3, 13, 14, 15, 16, 17, 18, 19, 20] #copy.deepcopy(robots[0]._urdf_joints) #[-2:]
    collision_radii = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    for i, collision_link_nr in enumerate(collision_keys_robot0):
         env.add_collision_link(0, collision_link_nr, shape_type='sphere', size=[collision_radii[i]])
    return (env, goal)


def set_planner(goal: GoalComposition, nr_obst: int = 0, degrees_of_freedom: int = 6):
    """
    Initializes the fabric planner for the kuka robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    robot_model = RobotModel('dingo_kinova', model_name='dingo_kinova')
    urdf_file = robot_model.get_urdf_path()
    with open(urdf_file, "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="base_link",
        end_links=["arm_end_effector_link"],
    )

    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    planner.load_fabrics_configuration(CONFIG_FABRICS)
    planner.load_problem_configuration(CONFIG_PROBLEM)
    planner.concretize()
    planner.export_as_c("controller_dinova.c")
    return planner


def run_kinova_example(n_steps=5000, render=True, dof=9):
    nr_obst = 2
    (env, goal) = initalize_environment(render, nr_obst=nr_obst)
    planner = set_planner(goal, nr_obst, degrees_of_freedom=dof)
    action = np.zeros(dof)
    ob, *_ = env.step(action)

    for w in range(n_steps):
        ob_robot = ob['robot_0']

        arguments_dict = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][nr_obst+2]['position'],
            weight_goal_0=ob_robot['FullSensor']['goals'][nr_obst+2]['weight'],
            x_obsts=[ob_robot['FullSensor']['obstacles'][nr_obst]['position'],
                     ob_robot['FullSensor']['obstacles'][nr_obst + 1]['position']],
            radius_obsts=[ob_robot['FullSensor']['obstacles'][nr_obst + 0]['size'],
                          ob_robot['FullSensor']['obstacles'][nr_obst + 1]['size']],
            radius_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst]['size'],
            radius_body_chassis_link=0.4,
            radius_body_arm_shoulder_link=0.1,
            radius_body_arm_end_effector_link = 0.1,
            radius_body_arm_upper_wrist_link = 0.1,
            radius_body_arm_lower_wrist_link = 0.1,
            radius_body_arm_forearm_link=0.1,
            constraint_0=np.array([0, 0, 1, 0.0])
        )

        action = planner.compute_action(**arguments_dict)
        ob, *_ = env.step(action)
    env.close()
    return {}


if __name__ == "__main__":
    dof = 9
    res = run_kinova_example(n_steps=5000, dof=dof)