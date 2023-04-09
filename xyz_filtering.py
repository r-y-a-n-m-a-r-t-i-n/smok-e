import sys
import json
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x, r_des):
    # origin is at arm base (so ground z = -h)
    # y is robot forward, x is robot right, z is up
    q = x

    theta_r = q[0]
    theta_s = q[1]
    theta_e = q[2]

    # first link length in meters - servo 1 axis to servo 2 axis
    L_1 = 0.458978
    # second link length in meters - servo 2 axis to gripper center
    L_2 = 0.563372

    r_x = (L_1 * np.cos(theta_s) + L_2 * np.cos(theta_s + theta_e)) * np.sin(theta_r)
    r_y = (L_1 * np.cos(theta_s) + L_2 * np.cos(theta_s + theta_e)) * np.cos(theta_r)
    r_z = L_1 * np.sin(theta_s) + L_2 * np.sin(theta_s + theta_e)

    return np.array((r_x, r_y, r_z)) - r_des


def find_arm_trajectory(r_des, q_guess):
    r_guess = f(q_guess, r_des)
    sol = scipy.optimize.fsolve(f, r_guess, args=r_des)

    return sol


def main(currPos: np.ndarray, finPos: np.ndarray):
    # where am I now
    r_curr = currPos

    # where is the trash
    r_des = finPos

    # TODO: intermediate positions between start and end - arc?
    # the linspace is just a line, change to arc with start and end on circle and servo 1 as center
    xyz_positions = np.linspace(r_curr, r_des)

    # guess angle positions for solver
    q_guesses = np.empty((len(xyz_positions), 3))
    for i in range(len(q_guesses)):
        # could try to come up with decent guess for each, but filtering outliers later so may not matter
        q_guess = np.array([
            0,  # theta_r guess
            np.pi / 2,  # theta_s guess
            0,  # theta_e guess
        ])
        q_guesses[i] = q_guess

    # desired joint positions and velocities
    joint_positions = np.empty((len(xyz_positions), 3))
    for i in range(len(joint_positions)):
        joint_positions[i] = find_arm_trajectory(xyz_positions[i], q_guesses[i])

    mean = np.mean(joint_positions, axis=0)
    # print("Mean: ", mean)
    standard_deviation = np.std(joint_positions, axis=0)
    distance_from_mean = abs(joint_positions - mean)
    # print("Std Dev: ", standard_deviation)
    # print("dist from mean: ", distance_from_mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    not_outlier_indices = np.where(np.all(not_outlier, axis=1))
    joint_positions_no_outliers = joint_positions[not_outlier_indices]

    # print(joint_positions_no_outliers)

    init = joint_positions[0]
    fin = joint_positions[-1]

    # print(init)
    # print(fin)

    joint_positions = np.vstack((init, joint_positions_no_outliers))
    joint_positions = np.vstack((joint_positions, fin))
    # print(joint_positions)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2])
    # ax.scatter(xyz_positions[:, 0], xyz_positions[:, 1], xyz_positions[:, 2])
    plt.show()

    # convert numpy array to dictionary
    d = dict(enumerate(joint_positions.tolist(), 1))

    # convert dictionary to json
    j = json.dumps(d)

    return j


if __name__ == "__main__":
    main(np.fromstring(sys.argv[1], dtype=float, sep=','), np.fromstring(sys.argv[2], dtype=float, sep=','))
