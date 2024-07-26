import numpy as np
import matplotlib.pyplot as plt

class PlotRollouts(object):
    def __init__(self, params):
        self.params = params
        self.dof = params.dof
        self.N_horizon = params.N_HORIZON
        self.dt = params.dt

        self.state_j = [ [] for _ in range(self.dof[0]) ]
        self.vel_j = [ [] for _ in range(self.dof[0]) ]
        self.acc_j = [ [] for _ in range(self.dof[0]) ]
        self.state_N_time_j = [ [] for _ in range(self.dof[0])]
        self.q_dot_N_time_j = [ [] for _ in range(self.dof[0])]
        self.q_ddot_N_time_j = [[] for _ in range(self.dof[0])]

    def update_variable_lists(self, q_pandas, qdot_pandas, action, w, forwardplanner, inputs_action):
        for df in range(self.dof[0]):
            self.state_j[df].append(q_pandas[0][df])
            self.vel_j[df].append(qdot_pandas[0][df])
            if self.params.fabrics_mode == "acc":
                self.acc_j[df].append(action[df])
            else:
                self.acc_j[df].append(0)

        if w % 100 == 0:
            [q_robots_N, q_dot_robots_N, q_ddot_robots_N] = forwardplanner.rollouts_numerical(inputs_action=inputs_action)
            # print(w)
            # for plotting rollouts
            # Forward Fabrics: Shows the inputs and positions over the horizon
            for df in range(self.dof[0]):
                self.state_N_time_j[df].append(q_robots_N["robot_0"][0][df])
                self.q_dot_N_time_j[df].append(q_dot_robots_N["robot_0"][0][df])
                self.q_ddot_N_time_j[df].append(q_ddot_robots_N["robot_0"][0][df])

    def plot_results(self): #, variables_plots):
        variables_plots = {"state_j": self.state_j, "vel_j": self.vel_j, "acc_j": self.acc_j,
                        "state_N_time_j": self.state_N_time_j,
                        "q_dot_N_time_j": self.q_dot_N_time_j,
                        "action_N_time_j": self.q_ddot_N_time_j,
                        "x_obsts": [], #x_obsts,
                        "r_obsts": []} #radius_obsts}
        self.N_steps = len(variables_plots["state_j"][0])
        time_x = np.linspace(0, self.N_steps * 0.01, self.N_steps)
        n_steps_lim = int(self.N_steps / 100)  # don't want to plot all steps
        i_robot = 0

        # plotting
        colorscheme = ['r', 'b', 'm', 'g', 'k', 'y', 'deeppink']

        if n_steps_lim > 2:
            n_plots_N = 3
        else:
            n_plots_N = n_steps_lim

        fig2, ax2 = plt.subplots(n_plots_N, 3, figsize=(20, 15))
        # Plots over N
        for i in range(n_plots_N):
            i_dist = n_steps_lim / n_plots_N
            i_plt = int(np.floor(i_dist * i))
            time_0_N = i_plt * 100
            plt.gca().set_prop_cycle(None)
            for df in range(self.dof[i_robot]):
                ax2[i, 1].plot(variables_plots["q_dot_N_time_j"][df][i_plt], color=colorscheme[df])
                ax2[i, 1].plot(variables_plots["vel_j"][df][time_0_N:time_0_N + self.N_horizon], 'x',
                               color=colorscheme[df])

                ax2[i, 2].plot(variables_plots["action_N_time_j"][df][i_plt], color=colorscheme[df])
                ax2[i, 2].plot(variables_plots["acc_j"][df][time_0_N:time_0_N + self.N_horizon], 'x',
                               color=colorscheme[df])

                ax2[i, 0].plot(variables_plots["state_N_time_j"][df][i_plt], color=colorscheme[df])
                ax2[i, 0].plot(variables_plots["state_j"][df][time_0_N:time_0_N + self.N_horizon], 'x',
                               color=colorscheme[df])
            ax2[i, 2].set_title("Action ($\ddot{q}$) over N, time=" + str(i_plt * 100 * self.dt))
            ax2[i, 2].set_xlabel('$i=1:N$')
            ax2[i, 2].set_ylabel('$\ddot{q}')
            ax2[i, 1].set_title("Velocities ($\dot{q}$) over N, time=" + str(i_plt * 100 * self.dt))
            ax2[i, 0].set_title("Joint pos ($q$) over N, time=" + str(i_plt * 100 * self.dt))
            ax2[i, 0].set_xlabel('$i=1:N$')
            ax2[i, 0].set_ylabel('$q$')
            ax2[i, 0].grid()
            ax2[i, 1].grid()
            ax2[i, 2].grid()
        plt.show()