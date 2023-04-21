import sys
import json
import numpy as np
import scipy.optimize
import matplotlib
import matplotlib.pyplot as plt
import math


def fun(x, r_des):
    # origin is at arm base (so ground z = -h)
    # x is forward, y is up
    # theta=0 is straight out
    q = x

    theta_s = q[0]
    theta_e = q[1]

    # first link length in meters - servo 1 axis to servo 2 axis
    L_1 = 0.458978
    # second link length in meters - servo 2 axis to gripper center
    # old/long
    # L_2 = 0.563372
    # new/short
    L_2 = 0.385572

    r_x = L_1 * np.cos(theta_s) + L_2 * np.cos(theta_s + theta_e)
    r_y = L_1 * np.sin(theta_s) + L_2 * np.sin(theta_s + theta_e)

    return np.array((r_x, r_y)) - r_des


def find_arm_trajectory(r_des, q_guess):
    r_guess = fun(q_guess, r_des)
    # print(scipy.optimize.fsolve(f, r_guess, args=r_des, full_output=True))
    # sol = scipy.optimize.fsolve(f, r_guess, args=r_des)
    if r_des[0] > 0:
        bounds = ((-np.inf, -np.inf), (np.inf, 0))
    else:
        bounds = ((-np.inf, 0), (np.inf, np.inf))

    print(r_guess)

    sol = scipy.optimize.least_squares(fun, q_guess, args=(r_des,), bounds=bounds)
    # print(sol["x"])
    q = sol["x"]
    return q


def main(currPos: np.ndarray, finPos: np.ndarray):
    currPos[0] = currPos[0] * math.pi / 180 - math.pi / 2  # 90 deg is straight forward for shoulder
    currPos[1] = currPos[1] * math.pi / 180 - math.pi  # 180 deg is straight forward for forearm (in shoulder frame)

    # where is the trash
    r_des = finPos

    # guess angle positions for solver
    q_guess = np.array([
        0,            # theta_s guess in degrees
        -math.pi      # theta_e guess in degrees
    ])

    # desired joint positions and velocities
    q_final = find_arm_trajectory(r_des, q_guess)

    # debugging plots
    L_1 = 0.458978
    # L_2 = 0.563372
    L_2 = 0.385572

    q_final = q_final

    x1 = L_1 * np.cos(q_final[0])
    y1 = L_1 * np.sin(q_final[0])

    x2 = L_1 * np.cos(q_final[0]) + L_2 * np.cos(q_final[0] + q_final[1])
    y2 = L_1 * np.sin(q_final[0]) + L_2 * np.sin(q_final[0] + q_final[1])

    arm1x = np.linspace(0, x1)
    arm1y = np.linspace(0, y1)

    arm2x = np.linspace(x1, x2)
    arm2y = np.linspace(y1, y2)

    plt.scatter(arm1x, arm1y, color='black')
    plt.scatter(arm2x, arm2y, color='gray')
    plt.scatter(r_des[0], r_des[1], color='purple')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()

    # change to degrees and correct frame/orientation
    q_final[0] = (q_final[0] + math.pi / 2) * 180 / math.pi
    q_final[1] = (q_final[1] + math.pi) * 180 / math.pi

    # convert numpy array to json
    j = json.dumps(q_final.tolist())

    return j


if __name__ == "__main__":
    print(main(np.fromstring(sys.argv[1], dtype=float, sep=','), np.fromstring(sys.argv[2], dtype=float, sep=',')))
