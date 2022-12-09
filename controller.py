import numpy as np
import scipy.optimize
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import System

from robot import Robot


def f(x, robot, r_des):
    # origin is at arm base (so ground z = -h)
    # y is robot forward, x is robot right, z is up

    theta_r = x[0]
    theta_s = x[1]
    theta_e = x[2]

    L_1 = robot.L_1
    L_2 = robot.L_2

    r_x = (L_1 * np.cos(theta_s) + L_2 * np.cos(theta_s + theta_e)) * np.sin(theta_r)
    r_y = (L_1 * np.cos(theta_s) + L_2 * np.cos(theta_s + theta_e)) * np.cos(theta_r)
    r_z = L_1 * np.sin(theta_s) + L_2 * np.sin(theta_s + theta_e)

    return np.array((r_x, r_y, r_z)) - r_des


def find_arm_trajectory(r_des, q_guess):
    robot = Robot()

    r_guess = robot.r_q(q_guess)

    sol = scipy.optimize.fsolve(f, r_guess, (robot, r_des))

    return sol





class CalculateTrajectory(System):
    def __init__(self):

        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModelFromFile("planar_walker.urdf")
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName("base").body_frame(),
            RigidTransform.Identity()
        )
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        r_des = np.array((1, 1, -1))

        q_guess = np.array([
            0.0,  # theta_r guess
            0.0,  # theta_s guess
            0.0,  # theta_e guess
        ])

        sol = find_arm_trajectory(r_des, q_guess)

        print(sol)

        return(sol)


if __name__ == "__main__":
    print(sol)
