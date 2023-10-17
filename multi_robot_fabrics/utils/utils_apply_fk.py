import numpy as np

def compute_x_obsts_dyn_0(q_robots, qdot_robots, x_collision_sphere_poses = None, nr_robots=2, fk_dict_spheres=[], nr_dyn_obsts = [0, 0]):
    """
    Construct the position of the dynamic obstacles from the environment
    and the velocity using the symbolic functions.
    """
    q = [[] for _ in range(nr_robots)]
    qdot = [[] for _ in range(nr_robots)]

    for i_robot in range(nr_robots):
        q[i_robot] = np.append(q_robots[i_robot], 0)
        qdot[i_robot] = np.append(qdot_robots[i_robot], 0)

    x_dyns_obsts = [[] for _ in range(nr_robots)]
    # x_dyns_obsts_new = [[] for _ in range(nr_robots)]
    v_dyns_obsts = [[] for _ in range(nr_robots)]
    # v_dyns_obsts_old = [[] for _ in range(nr_robots)]
    x_dyns_obsts_per_robot = [[] for _ in range(nr_robots)]

    for i_robot in range(nr_robots):
        i_other_robots = [i for i in range(nr_robots) if i != i_robot]
        x_dyns_obsts_per_robot[i_robot] = [x_sphere for key, x_sphere in x_collision_sphere_poses.items() if
                                            str(i_robot) in key[0]]
        for i_other_robot in i_other_robots:
            x_dyns_obsts[i_other_robot] = x_dyns_obsts[i_other_robot] + x_dyns_obsts_per_robot[i_robot]
            # x_dyns_obsts_new[i_robot] = np.vsplit(fk_dict_spheres[i_other_robot]["fk_fun"](q_robots[i_other_robot]).full().transpose(), dof+1)
            v_dyns_obsts[i_robot].extend(np.vsplit(fk_dict_spheres[i_other_robot]["vel_fun"](q[i_other_robot], qdot[i_other_robot]).full().transpose(), nr_dyn_obsts[i_robot]))

        # for i_other_robot in i_other_robots:
        #     for i_sphere in range(len(v_robots[i_other_robot])):
        #         v_dyns_obsts_old[i_robot] = v_dyns_obsts_old[i_robot] + [v_robots[i_other_robot][i_sphere]] * n_obst_per_link
    return x_dyns_obsts, v_dyns_obsts, x_dyns_obsts_per_robot

def compute_endeffector(q_robots, qdot_robots, fk_endeff, nr_robots = 2):
    """
    Compute the end-effector positions and velocities of the robot based on the forward kinematics.
    """
    x_robots_ee = []
    v_robots_ee = []
    for i_robot in range(nr_robots):
        x_robots_ee.append(fk_endeff[i_robot]["fk_fun_ee"](q_robots[i_robot]).full().transpose()[0])
        v_robots_ee.append(fk_endeff[i_robot]["vel_fun_ee"](q_robots[i_robot], qdot_robots[i_robot]).full().transpose()[0])
    return x_robots_ee, v_robots_ee