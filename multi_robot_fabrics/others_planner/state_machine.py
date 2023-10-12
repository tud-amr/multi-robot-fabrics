import numpy as np
import copy

class StateMachine(object):
    def __init__(self, start_goal, nr_robots, nr_blocks, fk_fun_ee, robot_types):
        self.nr_robots = nr_robots
        self.robot_types = robot_types
        self.state_machine_panda = 1
        self.nr_blocks_panda_success = 0
        self.nr_blocks_panda_failed = 0
        self.nr_blocks_panda = nr_blocks
        self.weight_panda_high = 2
        self.weight_panda_low = 0
        self.time_block_move_panda = 0
        self.time_start_move_panda = 0
        self.stop_time_panda = 0

        if self.robot_types[1] =="kinova":
            self.weight_robot2_high = 4
            self.dist_start_constant_robot2 = 0.03
            self.dist_block_constant_robot2 = 0.01
            self.constant_time_gripping2 = 1.3
        else:
            self.weight_robot2_high = 10
            self.dist_start_constant_robot2 = 0.03
            self.dist_block_constant_robot2 = 0.01
            self.constant_time_gripping2 = 1.3

        self.fk_fun_ee = fk_fun_ee
        self.gripper_panda = "open"
        self.gripper_robot2 = "open"
        self.goal = start_goal
        self.weight_goal = 2
        self.start_goal = start_goal
        self.q_panda_gripper_opened = np.array([0.04, 0.04])
        self.q_kinova_gripper_opened = np.array([0.96, 0.21, -0.96, -0.21])
        self.time_gripping_panda = 0

    def get_goal_robot(self):
        return self.goal

    def get_nr_blocks_picked(self):
        return self.nr_blocks_panda_success

    def get_success_rate(self):
        success_rate_robot1 = (self.nr_blocks_panda_success - self.nr_blocks_panda_failed)/(self.nr_blocks_panda)
        avg_success_rate = success_rate_robot1
        return avg_success_rate

    def get_x_ee(self, q_robot):
        x_ee = self.fk_fun_ee(q_robot)
        return x_ee

    def get_distance_ee_goal(self, q_robot, goal):
        x_ee = self.get_x_ee(q_robot=q_robot)
        distance_ee_goal_robot = np.linalg.norm(x_ee[:2] - goal[:2])
        return distance_ee_goal_robot

    def get_distance_ee_goal3(self, q_robot, goal):
        x_ee = self.get_x_ee(q_robot=q_robot)
        distance_ee_goal_robot = np.linalg.norm(x_ee - goal)
        return distance_ee_goal_robot

    def get_distance_ee_start(self, q_robot):
        x_ee = self.get_x_ee(q_robot=q_robot)
        distance_ee_start_robot = np.linalg.norm(x_ee - self.start_goal)
        return distance_ee_start_robot

    def get_gripper_action_panda(self, q_panda_gripper):
        distance_to_full_open_panda = np.linalg.norm(q_panda_gripper - self.q_panda_gripper_opened)
        action_gripper_panda = np.zeros(2)

        if self.gripper_panda == "close":
            action_gripper_panda = np.ones(2) * -0.05

        elif self.gripper_panda == "open" and distance_to_full_open_panda >0.005:
            for z in range(2):
                if q_panda_gripper[z] > self.q_panda_gripper_opened[z]:
                    action_gripper_panda[z] = -0.4
                else:
                    action_gripper_panda[z] = 0.4

        return action_gripper_panda

    def get_gripper_action_kinova(self, q_gripper):
        action_gripper_kinova = np.zeros(4)
        distance_to_full_opening_kinova = np.linalg.norm(q_gripper - self.q_kinova_gripper_opened)

        if self.gripper_robot2 == "close":  #this is closing the gripper
            # weight_goal_0_kinova = 0

             #check if limits are being exceeded
            q_gripper_low = [-0.10, -1.04, -0.97, -0.51]  #from manual kinova
            q_gripper_high = [0.97, 0.22, 0.10, 0.22] #from manual kinova

            gripper_list_low = q_gripper<q_gripper_low
            index_low = [i for i, x in enumerate(gripper_list_low) if x==True]
            gripper_list_high = q_gripper>q_gripper_high
            index_high = [i for i, x in enumerate(gripper_list_high) if x==True]

            action_gripper_kinova[0] = -4  #higher works a bit, but very aggressive
            action_gripper_kinova[1]  = -4
            action_gripper_kinova[2] = 4
            action_gripper_kinova[3] = 4
            for z in index_low:
                if z<2:
                    action_gripper_kinova[z] = 0.4
                else:
                    action_gripper_kinova[z] = 0.4
            #
            for z in index_high:
                if z<2:
                    action_gripper_kinova[z] = -0.4
                else:
                    action_gripper_kinova[z] = -0.4

        elif self.gripper_robot2 == "open" and distance_to_full_opening_kinova > 0.1:  #this is opening the gripper
            for z in range(4):
                if q_gripper[z]>self.q_kinova_gripper_opened[z]:
                    action_gripper_kinova[z] = -0.4
                else:
                    action_gripper_kinova[z] = 0.4
        else:
             action_gripper_kinova = np.zeros(4)
        return action_gripper_kinova

    def get_weight_goal0(self):
        return self.weight_goal

    def get_gripper_status(self):
        return self.gripper_panda, self.gripper_robot2

    def get_state_machine_panda(self, q_robot, q_robot_gripper, goal_block, robot_type):
        goal_pregrasp = copy.deepcopy(goal_block)
        goal_pregrasp[2] = goal_pregrasp[2] + 0.1
        distance_ee_start_panda = self.get_distance_ee_start(q_robot)
        distance_ee_pregrasp_panda = self.get_distance_ee_goal(q_robot, goal=goal_pregrasp)
        distance_ee_block_panda = self.get_distance_ee_goal3(q_robot, goal=goal_block)
        distance_to_full_opening_panda = np.linalg.norm(q_robot_gripper - self.q_panda_gripper_opened)
        # stop if all blocks are picked
        if self.nr_blocks_panda_success > self.nr_blocks_panda-1:
            self.state_machine_panda = 10
        elif goal_block[2]<0.6:
            print("the block has been dropped!")
            self.nr_blocks_panda_success = self.nr_blocks_panda_success + 1  #to make sure that you continue grasping
            self.nr_blocks_panda_failed = self.nr_blocks_panda_failed + 1
            self.state_machine_panda  = 0

        # go to start position and open the gripper
        if self.state_machine_panda == 0:
            self.goal = self.start_goal
            self.gripper_panda = "open"
            if distance_ee_start_panda < 0.05:
                # print("starting position of the panda is reached")
                self.state_machine_panda = 1

        # move from the start position to the pregrasp position above the block
        elif self.state_machine_panda == 1:
            self.goal = goal_pregrasp
            if distance_ee_pregrasp_panda < 0.013:
                print("panda has reached the pregrasp position, it is now going go the grasp position")
                self.state_machine_panda = 2

        elif self.state_machine_panda == 2: #from pregrasp to picking the block
            self.goal = goal_block
            if distance_ee_block_panda < 0.013:
                print("panda has reached the goal position, it is now going to grap the block")
                self.gripper_panda = "close"
                self.weight_goal = self.weight_panda_low
                self.state_machine_panda = 3

        elif self.state_machine_panda == 3: #grasp the block
            self.goal = goal_block
            self.goal_above_block = copy.deepcopy(goal_block)
            self.goal_above_block[2] += 0.15  #todo: I think this should be increased to have an effect
            self.time_gripping_panda = self.time_gripping_panda + 1
            if self.time_gripping_panda > 0.3/0.01:
                self.time_gripping_panda = 0
                self.goal= self.start_goal
                self.weight_goal = self.weight_panda_high
                self.state_machine_panda = 12

        elif self.state_machine_panda == 12:
            self.goal = self.goal_above_block
            if self.get_distance_ee_goal(q_robot, goal=self.goal) <0.04:
                self.state_machine_panda = 4

        elif self.state_machine_panda == 4: #go to the starting position
            self.goal = self.start_goal
            if distance_ee_start_panda < 0.15:
                print("panda has reached the start position, it is now going to release the block")
                self.state_machine_panda = 5
                self.gripper_panda = "open"
                self.time_start_move_panda = 0

        elif self.state_machine_panda == 5: #release the block
            if distance_to_full_opening_panda < 0.005:
                self.state_machine_panda = 0
                self.nr_blocks_panda_success = self.nr_blocks_panda_success + 1
                self.goal = self.start_goal

        elif self.state_machine_panda == 10:
            if self.stop_time_panda == 0:
                print("panda has picked all blocks!")
                self.stop_time_panda = 1
                self.state_machine_panda == 0

        else:
            print("state provided is not feasible")
        # print("state_machine_panda", self.state_machine_panda)
        weight_goals = self.get_weight_goal0()
        q_dot_gripper_panda = self.get_gripper_action_panda(q_panda_gripper=q_robot_gripper)

        return self.state_machine_panda
