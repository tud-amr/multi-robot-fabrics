# needs to be moved somewhere else. Possibly to fabrics?
import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
import forwardkinematics.urdfFks.casadiConversion.geometry.transformation_matrix as T

class UtilsKinematics(object):
    """
    Class for forward kinematics using the provided URDF, symbolical Casadi variables and the forwardkinematics package.
    """
    def __init__(self):
        self.fk_sym_list = []
        self.jac_sym_list = []
        self.jac_dot_sym_list = []

    def necessary_kinematics(self, planner, center= False, i_robot=0):
        fk_sym = {}
        jac_sym = {}
        jac_dot_sym = {}
        fk_fun = {}
        jac_fun = {}
        jac_dot_fun = {}

        q = planner.variables.position_variable()
        qdot = planner.variables.velocity_variable()
        # self.transformation_to_link_center(planner, planner._forward_kinematics.robot._link_names, q)

        Jdot_sign = -1
        links = planner._forward_kinematics.robot._link_names
        index_end_link = links.index("panda_leftfinger")
        for i, link in enumerate(links[0:index_end_link]):
            if center:
                fk_link_i= planner._forward_kinematics._fks_center_T[link]  #doesn't work for all links!!
            else:
                fk_link_i = planner.get_forward_kinematics(links[i])
            jac_link_i = ca.jacobian(fk_link_i, q)
            jac_dot_link_i = Jdot_sign*ca.jacobian(ca.mtimes(jac_link_i, qdot), q)

            fk_sym[link] = fk_link_i
            jac_sym[link] = jac_link_i
            jac_dot_sym[link] = jac_dot_link_i

            fk_fun_link_i = ca.Function("fk_"+str(link), [q], [fk_link_i])
            jac_fun_link_i = ca.Function("jac_"+str(link), [q], [jac_link_i])
            jac_dot_fun_link_i = ca.Function("jac_dot_"+str(link), [q, qdot], [jac_dot_link_i])

            fk_fun[link] = fk_fun_link_i
            jac_fun[link] = jac_fun_link_i
            jac_dot_fun[link] = jac_dot_fun_link_i

        self.fk_sym_list.append(fk_sym)
        self.jac_sym_list.append(jac_sym)
        self.jac_dot_sym_list.append(jac_dot_sym)
        return fk_fun, jac_fun, jac_dot_fun


    def get_symbolic_kinematics(self) -> (list, list, list):
        return (self.fk_sym_list, self.jac_sym_list, self.jac_dot_sym_list)

    def define_forward_kinematics(self, planners, collision_links_nrs, collision_links):
        """
        Forward kinematics along the kinematic chain of the robots.
        This function basically calls 'necessary-kinematics' and places it in a dictionary
        """

        # utils_kinematics = utils.UtilsKinematics()
        # collision_links_nrs = parameters[4]
        # collision_links = parameters[5]
        nr_robots = len(collision_links_nrs)
        self.nr_robots = nr_robots
        fk_dict = {"fk_fun_center":[[] for _ in range(nr_robots)], "jac_fun_center":[[] for _ in range(nr_robots)], "jac_dot_fun_center":[[] for _ in range(nr_robots)],
                   "fk_fun":[[] for _ in range(nr_robots)], "jac_fun":[[] for _ in range(nr_robots)], "jac_dot_fun":[[] for _ in range(nr_robots)]}
        for i_robot in range(nr_robots):
            for center in [False]:
                fk_fun, jac_fun, jac_dot_fun = self.necessary_kinematics(planners[i_robot], center=center)
                fk_sym_list, jac_sym_list, jac_dot_sym_list = self.get_symbolic_kinematics()
                if center == True:
                    str_name = "_center"
                else:
                    str_name = ""
                for link in collision_links[i_robot]:
                    fk_dict["fk_fun" + str(str_name)][i_robot].append(fk_fun[link])
                    fk_dict["jac_fun" + str(str_name)][i_robot].append(jac_fun[link])
                    fk_dict["jac_dot_fun" + str(str_name)][i_robot].append(jac_dot_fun[link])
        return fk_dict

    def define_symbolic_collision_link_poses(self, urdf_files, collision_links, sphere_transformations, n_obst_per_link=1, mount_transform=[]):
        # q = planner.variables.position_variable()
        # qdot = planner.variables.velocity_variable()
        nr_robots = len(sphere_transformations)
        self.nr_robots = nr_robots
        fk_spheres = [{"fk_fun":[], "vel_fun":[]} for _ in range(nr_robots)]
        fk_spheres_sym = [{"fk_sym":[], "vel_sym":[]} for _ in range(nr_robots)]

        urdf_link = urdf_files["URDF_file_panda"]
        with open(urdf_link, "r") as file:
            urdf = file.read()

        fk = GenericURDFFk(urdf, rootLink="panda_link0", end_link="panda_leftfinger")
        n = fk.n()
        q_ca = ca.SX.sym("q", n)
        qdot_ca = ca.SX.sym("qdot", n)
        for i_robot in range(nr_robots):
            fk.set_mount_transformation(mount_transformation=mount_transform[i_robot])
            for i_link, link in enumerate(collision_links[i_robot]):
                for i_sphere in range(n_obst_per_link):
                    fk_spheres_i = fk.fk(q_ca, parent_link="panda_link0", child_link=link,
                                      link_transformation=sphere_transformations[i_robot][i_link][i_sphere],
                                      positionOnly=True)
                    vel_sphere_i = ca.jacobian(fk_spheres_i, q_ca) @ qdot_ca
                    fk_spheres_sym[i_robot]["fk_sym"].append(fk_spheres_i)
                    fk_spheres_sym[i_robot]["vel_sym"].append(vel_sphere_i)
            fk_spheres[i_robot]["fk_fun"] = ca.Function("fk_spheres_robot"+str(i_robot), [q_ca], [ca.hcat(fk_spheres_sym[i_robot]["fk_sym"])], ["q"], ["xyz_spheres"])
            fk_spheres[i_robot]["vel_fun"] = ca.Function("vel_spheres_robot" + str(i_robot), [q_ca, qdot_ca],
                                               [ca.hcat(fk_spheres_sym[i_robot]["vel_sym"])], ["q", "qdot"], ["vel_spheres"])

        self.fk_spheres_sym = fk_spheres_sym
        self.q_ca = q_ca
        self.qdot_ca = qdot_ca
        return fk_spheres

    def define_symbolic_endeffector(self, planners):
        # --- initialize symbolic forward kinematics ---#
        fk_endeff = [{"fk_fun_ee": [], "vel_fun_ee": []} for _ in range(self.nr_robots)]
        q_pandas_sym = [[] for _ in range(self.nr_robots)]
        qdot_pandas_sym = [[] for _ in range(self.nr_robots)]
        for i_robot in range(self.nr_robots):
            q_pandas_sym[i_robot] = planners[i_robot].variables.position_variable()
            qdot_pandas_sym[i_robot] = planners[i_robot].variables.velocity_variable()

            endeff_panda = planners[i_robot].get_forward_kinematics("panda_hand")
            endeff_panda_vel = ca.jacobian(endeff_panda, q_pandas_sym[i_robot]) @ qdot_pandas_sym[i_robot]

            fk_endeff[i_robot]["fk_fun_ee"] = ca.Function("fk_endeff", [q_pandas_sym[i_robot]], [endeff_panda])
            fk_endeff[i_robot]["vel_fun_ee"] = ca.Function("fk_endeff_vel", [q_pandas_sym[i_robot], qdot_pandas_sym[i_robot]], [endeff_panda_vel])

        return fk_endeff
        # ---- new method ---- #
        # urdf_link = urdf_files["URDF_file_panda"]
        # with open(urdf_link, "r") as file:
        #     urdf = file.read()
        #
        # fk_endeff = [{"fk_fun_ee": [], "vel_fun_ee": []} for _ in range(self.nr_robots)]
        # fk = GenericURDFFk(urdf, rootLink="panda_link0", end_link="panda_hand")
        # n = fk.n()
        # q_ca = ca.SX.sym("q", n)
        # qdot_ca = ca.SX.sym("qdot", n)
        # for i_robot in range(self.nr_robots):
        #     fk.set_mount_transformation(mount_transformation=mount_transform[i_robot])
        #     fk_endeff[i_robot] = fk.fk(q_ca, parent_link="panda_link0", child_link="panda_hand", positionOnly=True)

    def return_symbolic_collision_links(self):
        return self.fk_spheres_sym

    # def x_dyn_obsts_functions(self):
    #     """
    #     Returns function and symbolic struct of all dynamic obstacles considered (in a multirobot scenario)
    #     Uses all previously constructed information
    #     """
    #     fk_dyn_obsts_fun = [{"fk_fun":[], "vel_fun":[]} for _ in range(self.nr_robots)]
    #     fk_dyn_obsts_sym = [{"fk_sym":[], "vel_sym":[]} for _ in range(self.nr_robots)]
    #
    #     for i_robot in range(self.nr_robots):
    #         i_other_robots = [i for i in range(self.nr_robots) if i != i_robot]
    #         for i_other_robot in i_other_robots:
    #             fk_dyn_obsts_sym[i_robot]["fk_sym"].append(self.fk_spheres_sym[i_other_robot])
    #
    #         fk_dyn_obsts_fun[i_robot]["fk_fun"] = ca.Function("fk_dyn_obsts" + str(i_robot), [self.q_ca],
    #                                                     [ca.hcat(fk_dyn_obsts_sym[i_robot]["fk_sym"])], ["q"],
    #                                                     ["xyz_dyn_obsts"])
    #         fk_dyn_obsts_fun[i_robot]["vel_fun"] = ca.Function("vel_dyn_obsts" + str(i_robot), [self.q_ca, self.qdot_ca],
    #                                                      [ca.hcat(fk_dyn_obsts_sym[i_robot]["vel_sym"])], ["q", "qdot"],
    #                                                      ["vel_dyn_obsts"])
    #
    #     return fk_dyn_obsts_fun
        # self.n_obst_per_link  = n_obst_per_link
        # nr_robots = len(collision_links)
        # sphere_names = [[] for _ in range(nr_robots)]
        # x_obsts_dyn_symbolic = [[] for _ in range(nr_robots)]
        # fk_sphere_dict = {"fk_fun": [{} for _ in range(nr_robots)],
        #            "jac_fun": [{} for _ in range(nr_robots)],
        #            "jac_dot_fun": [{} for _ in range(nr_robots)]}
        #
        # for i_robot in range(nr_robots):
        #     for i_link, link in enumerate(collision_links[i_robot]):
        #         for i_sphere in range(self.n_obst_per_link):
        #             fk_sphere = self.fk_sym_list[i_robot][link] + sphere_transformations[i_robot][i_link][i_sphere][0:3, 3]
        #             sphere_names[i_robot].append('sphere'+str(i_robot)+"_"+str(i_link)+"_"+str(i_sphere))
        #             x_obsts_dyn_symbolic[i_robot].append(fk_sphere)
        #     q = planners[i_robot].variables.position_variable()
        #     fk_sphere_dict["fk_fun"][i_robot] = ca.Function("fk_spheres", [q], x_obsts_dyn_symbolic[i_robot])
        # return {} #fk_sphere_dict
