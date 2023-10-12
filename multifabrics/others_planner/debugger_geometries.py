import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from matplotlib import pyplot as plt

class DebuggerGeometries(object):
    def __init__(self, nr_robots=2, n_steps=1000):
        self.nr_robots = nr_robots
        self.dof = [7, 7]
        self.leaves_struct = [[] for i_robot in range(self.nr_robots)]
        self.leaf_names = [[] for i_robot in range(self.nr_robots)]
        self.debug_evaluations = [{"goal_geometries":[[] for df in range(self.dof[0])],
                                   "obst_geometries":[[] for df in range(self.dof[0])],
                                   "limit_geometries":[[] for df in range(self.dof[0])],
                                   "constraint_geometries":[[] for df in range(self.dof[0])],
                                   "predicted_actions":[[] for df in range(self.dof[0])]}
                                  for i_robot in range(self.nr_robots)]
        self.actions_list = [[[] for df in range(self.dof[0])], [[] for df in range(self.dof[0])]]
        self.n_steps = n_steps

    def concretize_leaves(self, planners):
        for i_robot in range(self.nr_robots):
            leaves = []
            self.leaf_names[i_robot] = list(planners[i_robot].leaves.keys())
            leaves_struct = planners[i_robot].get_leaves(leaf_names=self.leaf_names[i_robot])
            for i_leaf, leaf in enumerate(leaves_struct):
                leaves.append(leaf.concretize())
            self.leaves_struct[i_robot] = leaves_struct

    def evaluate_geometries(self, arguments, actions):
        for i_robot in range(self.nr_robots):
            debug_evaluation_goal = []
            debug_evaluation_obst = []
            debug_evaluation_limit = []
            debug_evaluation_constraint = []
            for i_leaf, leaf in enumerate(self.leaves_struct[i_robot]):
                leaf_name = self.leaf_names[i_robot][i_leaf]
                debug_evaluation = np.abs(leaf.evaluate(**arguments[i_robot])["h_pulled"])
                if "goal" in leaf_name:
                    debug_evaluation_goal.append(debug_evaluation)
                elif "obst" in leaf_name:
                    debug_evaluation_obst.append(debug_evaluation)
                elif "limit" in leaf_name:
                    debug_evaluation_limit.append(debug_evaluation)
                elif "constraint" in leaf_name:
                    debug_evaluation_constraint.append(debug_evaluation)
                else:
                    print("Unknown geometry")
            debug_evaluation_goal_sum = sum(debug_evaluation_goal)
            debug_evaluation_obst_sum = sum(debug_evaluation_obst)
            debug_evaluation_limit_sum = sum(debug_evaluation_limit)
            debug_evaluation_constraint_sum = sum(debug_evaluation_constraint)
            predicted_action = sum([debug_evaluation_goal_sum, debug_evaluation_obst_sum, debug_evaluation_limit_sum, debug_evaluation_constraint_sum])
            for df in range(self.dof[i_robot]):
                self.debug_evaluations[i_robot]["goal_geometries"][df].append(debug_evaluation_goal_sum[df])
                self.debug_evaluations[i_robot]["obst_geometries"][df].append(debug_evaluation_obst_sum[df])
                self.debug_evaluations[i_robot]["limit_geometries"][df].append(debug_evaluation_limit_sum[df])
                self.debug_evaluations[i_robot]["constraint_geometries"][df].append(debug_evaluation_constraint_sum[df])
                self.debug_evaluations[i_robot]["predicted_actions"][df].append(predicted_action[df])
                self.actions_list[i_robot][df].append(actions[i_robot][df]/0.01)

    def return_debug_evaluations(self):
        return self.debug_evaluations

    def plot_debug_evaluations(self, ratio_steps=1):
        time_x = np.linspace(0, self.n_steps * 0.01, int(self.n_steps/ratio_steps))
        fig, ax = plt.subplots(2, 7, figsize=(20, 15))
        for i_robot in range(self.nr_robots):
            for df in range(self.dof[i_robot]):
                ax[i_robot, df].plot(time_x, self.debug_evaluations[i_robot]["obst_geometries"][df])
                ax[i_robot, df].plot(time_x, self.debug_evaluations[i_robot]["limit_geometries"][df])
                ax[i_robot, df].plot(time_x, self.debug_evaluations[i_robot]["constraint_geometries"][df])
                ax[i_robot, df].plot(time_x, self.debug_evaluations[i_robot]["goal_geometries"][df])
                # ax[i_robot, df].plot(time_x, self.actions_list[i_robot][df])
                # ax[i_robot, df].plot(time_x, self.debug_evaluations[i_robot]["predicted_actions"][df])
                ax[i_robot, df].grid()
                ax[i_robot, df].legend(['obst', 'limit', 'constraints', "goal", "action", "predicted action"])

        plt.show()
        kkk=1
        # ax[i, 1].plot(variables_plots["q_dot_N_time_j"][df][i_plt], color=colorscheme[df])
