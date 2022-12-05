import scipy.optimize

from robot import Robot
import numpy as np
import scipy as sp


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


r_des = np.array((1, 1, -1))

q_guess = np.array([
    -np.pi / 4.0,  # theta_r guess
    -np.pi / 4.0,  # theta_s guess
    -np.pi / 4.0,  # theta_e guess
])

sol = find_arm_trajectory(r_des, q_guess)

print(sol)

