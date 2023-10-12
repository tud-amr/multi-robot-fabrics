import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

class generate_plots(object):

    def __init__(self, dof, N_steps, dt, fabrics_mode, dt_MPC=0.01):
        self.dof = dof
        self.time_x = np.linspace(0, N_steps * dt, N_steps)
        self.n_steps_lim = int(N_steps / 100)  # don't want to plot all steps
        self.fabrics_mode = fabrics_mode
        self.dt_MPC = dt_MPC

    def plot_results(self, variables_plots, nr_obst, r_obst = 0.1, MPC_LAYER=True):
        # Plots
        fig = plt.figure(figsize=(20, 15))
        plt.clf()
        gs = GridSpec(3, 3, figure=fig)

        # plotting
        colorscheme = ['r', 'b', 'm', 'g', 'k', 'y', 'deeppink']
        if self.fabrics_mode == "acc":
            str_mode = "$\ddot{q}$"
        else:
            str_mode = "$\dot{q}$"

        # plot trajectory
        ax = fig.add_subplot(gs[0:2, 0:2], projection='3d')

        #plot sphere
        x_obsts = variables_plots["x_obsts"]
        for x_obst in x_obsts:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = r_obst * np.outer(np.cos(u), np.sin(v))+x_obst[0]
            y = r_obst * np.outer(np.sin(u), np.sin(v))+x_obst[1]
            z = r_obst * np.outer(np.ones(np.size(u)), np.cos(v))+x_obst[2]
            # Plot the surface
            ax.plot_surface(x, y, z)
        ax.plot3D(variables_plots["pos_xyz"][0], variables_plots["pos_xyz"][1], variables_plots["pos_xyz"][2])
        # Set an equal aspect ratio
        ax.set_aspect('equal')
        plt.xlim([0, 2])
        plt.ylim([-1, 1])
        ax.set_zlim([0, 2])
        plt.grid()

        plt.title("($x$, $y$, $z$)-position end-effector")
        plt.xlabel('$x$ [m]')
        plt.ylabel('$y$ [m]')
        ax.set_zlabel('$z$ [m]')
        plt.grid()

        fig.add_subplot(gs[0, 2])
        for df in range(self.dof):
            q_plt = plt.plot(self.time_x, variables_plots["state_j"][df], colorscheme[df])
        plt.title("Joint position ($q$)")
        plt.xlabel('time [s]')
        plt.ylabel('$q$ [rad]')
        plt.grid()

        fig.add_subplot(gs[1, 2])
        for df in range(self.dof):
            qdot_plt = plt.plot(self.time_x, variables_plots["vel_j"][df], colorscheme[df])
        plt.title("Joint velocity ($\dot{q}$")
        plt.xlabel('time [s]')
        plt.ylabel('$\dot{q}$ [rad/s]')
        plt.grid()

        fig.add_subplot(gs[2, 2])
        for df in range(self.dof):
            qddot_plot = plt.plot(self.time_x, variables_plots["acc_j"][df], colorscheme[df])
        plt.title("Joint acceleration ($\ddot{q}$)")
        plt.xlabel('time [s]')
        plt.ylabel('$\ddot{q}$ [rad/s^2]')
        plt.grid()

        fig.add_subplot(gs[2, 0])
        exit_plt = plt.plot(self.time_x, variables_plots["solver_time"])
        plt.xlabel('time [s]')
        plt.ylabel('solver time [s]')
        plt.title("Solver time")
        plt.grid()

        fig.add_subplot(gs[2, 1])
        if MPC_LAYER == True:
            comp_plt = plt.plot(self.time_x, variables_plots["exitflag"])
        plt.xlabel('time [s]')
        plt.ylabel('exitflag [-]')
        plt.title("Exitflag")
        plt.grid()

        plt.show()
        if self.n_steps_lim > 2:
            n_plots_N = 3
        else:
            n_plots_N = self.n_steps_lim

        if n_plots_N == 1:
            n_plotting_N = 2
        else:
            n_plotting_N = n_plots_N

        if self.fabrics_mode == "acc":
            fig2, ax2 = plt.subplots(n_plotting_N, 3, figsize=(20, 15))
        else:
            fig2, ax2 = plt.subplots(n_plotting_N, 2, figsize=(20, 15))
    # Plots over N
        for i in range(n_plots_N):
            i_dist = self.n_steps_lim/n_plots_N
            i_plt = int(np.floor(i_dist*i))
            plt.gca().set_prop_cycle(None)
            for df in range(self.dof):
                if self.fabrics_mode == "acc":
                    ax2[i, 2].plot(variables_plots["qddot_N_time_j"][df][i_plt], color=colorscheme[df])
                    if MPC_LAYER == True:
                        ax2[i, 2].plot(variables_plots["qddot_MPC_N_time_j"][df][i_plt], "x", color=colorscheme[df])
                    ax2[i, 2].set_title("Joint acceleration over N, time=" + str(i_plt * 100 * self.dt_MPC))
                    ax2[i, 2].set_xlabel('$i=1:N$')
                    ax2[i, 2].set_ylabel("$\ddot{q}$")
                    ax2[i, 2].grid()

                ax2[i, 1].plot(variables_plots["qdot_N_time_j"][df][i_plt], color=colorscheme[df])
                if MPC_LAYER == True:
                    ax2[i, 1].plot(variables_plots["qdot_MPC_N_time_j"][df][i_plt], "x", color=colorscheme[df])

                ax2[i, 0].plot(variables_plots["q_N_time_j"][df][i_plt], color=colorscheme[df])
                if MPC_LAYER == True:
                    ax2[i, 0].plot(variables_plots["q_MPC_N_time_j"][df][i_plt],  "x", color=colorscheme[df])
            ax2[i, 1].set_title("Joint velocity over N, time="+ str(i_plt*100*self.dt_MPC));  ax2[i, 1].set_xlabel('$i=1:N$'); ax2[i, 1].set_ylabel("$\dot{q}$")
            ax2[i, 0].set_title("Joint pos ($q$) over N, time="+ str(i_plt*100*self.dt_MPC));  ax2[i, 0].set_xlabel('$i=1:N$'); ax2[i, 0].set_ylabel('$q$')
            ax2[i, 0].grid()
            ax2[i, 1].grid()
        plt.show()
