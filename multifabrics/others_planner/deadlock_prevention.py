import numpy as np
import itertools

class deadlockprevention(object):

    def __init__(self, dof, n_robots, N_horizon):
        self.dof = dof
        self.n_robots = n_robots
        self.N_horizon = N_horizon
        self.i_leader = 0
        self.i_follower = 1
        if self.dof[0] == 2:  #for pointmass
            self.avg_vel_constant = 0.03
            self.dist_constant = 1
            self.goal_weight_follower = 10
            self.goal_weight_leader = 1
            self.time_wait = 50
            self.nr_goal_scale = 100
            self.goal_robot0 = np.array([0, 0])
        else:
            self.avg_vel_constant = 0.16
            self.dist_constant = 0
            self.goal_weight_follower = 2
            self.goal_weight_leader = 3
            self.time_wait = 300
            self.nr_goal_scale = 2
            self.goal_robot0 = np.array([0, 0, 0])

        robot_nrs = list(range(self.n_robots))
        self.robot_combinations = list(itertools.combinations(robot_nrs, 2))
        self.deadlock_robots = [0]*self.n_robots
        self.deadlock_combinations = [0]*(len(self.robot_combinations))
        self.i_robots_dead = [0, 1]
        self.time_in_deadlock = 0

    def compute_velocity_average(self, q_dot_robots_N):
        avg_sum = 0
        for i_robot in range(self.n_robots):
            for df in range(self.dof[i_robot]):
                q_dot_N = q_dot_robots_N["robot_"+str(i_robot)][df]
                q_dot_N_squared = [np.sqrt(q_dot_N_i**2) for q_dot_N_i in q_dot_N]
                avg_sum = avg_sum + sum(q_dot_N_squared)/(self.N_horizon*self.dof[i_robot])
        return avg_sum

    def compute_distance_to_goal(self, x_robot, goal_robot):
        dist_to_goal = np.linalg.norm(x_robot - goal_robot)
        return dist_to_goal


    def deadlock_checking(self, x_robots, goal_robots, goal_weights, time_step, time_deadlock_out, avg_sum, state_machine_robots = []):
        # give priority to the one that is slightly closer to the goal, otherwise random.
        # avg_sum = self.compute_velocity_average(q_dot_robots_N)
        # check_dist_endeff_list = []
        deadlock = False

        #check which robot combinations are in deadlock
        deadlock_distance = [100 for _ in range(len(self.robot_combinations))]
        dist_robot_to_goal = [np.linalg.norm(x_robots[i_robot] - goal_robots[i_robot]) for i_robot in range(self.n_robots)]

        for z, i_robots in enumerate(self.robot_combinations):
            dist_to_goal_sum = self.compute_distance_to_goal(x_robots[i_robots[0]], goal_robots[i_robots[0]]) + self.compute_distance_to_goal(x_robots[i_robots[1]], goal_robots[i_robots[1]])
            check_state = (state_machine_robots[i_robots[0]] in [0, 1]) and (state_machine_robots[i_robots[1]] in [0, 1]) #todo: add state 3????
            dist_endeff = np.linalg.norm(x_robots[i_robots[0]] - x_robots[i_robots[1]])
            check_dist_endeff = dist_endeff < 0.35

            if avg_sum < self.avg_vel_constant and dist_to_goal_sum>self.dist_constant and time_step>10 and check_state and check_dist_endeff:
                for i_robot in i_robots:
                    self.deadlock_robots[i_robot] = self.deadlock_robots[i_robot] + 1
                    self.deadlock_combinations[z] = self.deadlock_combinations[z] + 1
                    deadlock_distance[z] = dist_endeff

                if len(self.deadlock_combinations) >0:
                    deadlock = True
                    deadlock_min_dist = 100 #.index(min(deadlock_distance))

                    for z, deadlock_combi in enumerate(self.deadlock_combinations):
                        if deadlock_distance[z] < deadlock_min_dist:
                            deadlock_min_dist = deadlock_distance[z]
                            index_min_dist = z
                            self.i_robots_dead = list(self.robot_combinations[z])

        #pick the deadlock combination which you are going to resolve:
        if deadlock == True and time_step>10:
            # give priority to the one that is slightly closer to the goal, otherwise robot 0 is the follower
            if dist_robot_to_goal[self.i_robots_dead[0]] > dist_robot_to_goal[self.i_robots_dead[1]]:
                self.i_leader = self.i_robots_dead[1]
                self.i_follower = self.i_robots_dead[0]
            else:
                self.i_leader = self.i_robots_dead[0]
                self.i_follower = self.i_robots_dead[1]

            #compute distance and change weights and goal of the follower
            diff_robot_pos = x_robots[self.i_leader] - x_robots[self.i_follower]
            diff_goal = diff_robot_pos * self.nr_goal_scale
            if np.linalg.norm(diff_goal)>0.05:
                self.goal_robot0 = x_robots[self.i_follower] - 0.3/np.linalg.norm(diff_goal) * diff_goal
            else:
                self.goal_robot0 = x_robots[self.i_follower] - diff_robot_pos * self.nr_goal_scale
            if self.goal_robot0[2] < 0:
                self.goal_robot0[2] = 0.1
            # print("deadlock has appeared at time:", time_step * 0.01, "robot ", str(self.i_leader), "is the leader (has priority)")
            goal_weights[self.i_leader] = self.goal_weight_leader
            goal_weights[self.i_follower] = self.goal_weight_follower
            goal_robots[self.i_follower] = self.goal_robot0
            self.time_in_deadlock = self.time_in_deadlock+1
            time_deadlock_out = 0

        elif state_machine_robots[self.i_robots_dead[0]] == 2 or state_machine_robots[self.i_robots_dead[1]] == 2:
            time_deadlock_out = 400

        elif time_deadlock_out < self.time_wait:
            goal_weights[self.i_leader] = self.goal_weight_leader
            goal_weights[self.i_follower] = self.goal_weight_follower
            goal_robots[self.i_follower] = self.goal_robot0
            time_deadlock_out = time_deadlock_out + 1


        return goal_robots, goal_weights, time_deadlock_out