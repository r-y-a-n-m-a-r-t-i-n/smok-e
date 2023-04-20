import sys
import json
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def f(x, r_des):
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
    L_2 = 0.563372
    # new/short

    r_x = L_1 * np.cos(theta_s) + L_2 * np.cos(theta_s + theta_e)
    r_y = L_1 * np.sin(theta_s) + L_2 * np.sin(theta_s + theta_e)

    return np.array((r_x, r_y)) - r_des

def find_arm_trajectory(r_des, q_guess):
    r_guess = f(q_guess, r_des)
    print(scipy.optimize.fsolve(f, r_guess, args=r_des, full_output=True))
    sol = scipy.optimize.fsolve(f, r_guess, args=r_des)

    return sol


def calcXcircle(y, r, inFront):
    x = np.sqrt(r ** 2 - y ** 2)
    x_neg = -x

    if inFront:
        return x, y
    else:
        return x_neg, y


def main(currPos: np.ndarray, finPos: np.ndarray):
    # where am I now
    r_curr = f(currPos, 0)
    r = np.linalg.norm(r_curr)

    # where is the trash
    r_des = finPos

    xy_positions = np.linspace(r_curr, r_des)

    # only use the circle path if depositing trash/resetting arm (when arm must go over robot)
    # otherwise travel in linear path
    if np.sign(r_des[0]) != np.sign(r_curr[0]):
        for i in range(len(xy_positions)):
            xy_positions[i] = calcXcircle(xy_positions[i][1], r, xy_positions[i][0] >= 0)

    # guess angle positions for solver
    q_guesses = np.empty((len(xy_positions), 2))
    for i in range(len(q_guesses)):
        # could try to come up with decent guess for each, but filtering outliers later so may not matter
        if xy_positions[i][0] < 0:
            q_guess = np.array([
                0,  # theta_s guess
                -np.pi / 2  # theta_e guess
            ])
        else:
            q_guess = np.array([
                0,  # theta_s guess
                -np.pi / 2,  # theta_e guess
            ])
        q_guesses[i] = q_guess

    # desired joint positions and velocities
    joint_positions = np.empty((len(xy_positions), 2))
    for i in range(len(joint_positions)):
        joint_positions[i] = find_arm_trajectory(xy_positions[i], q_guesses[i])

    # filter outlier positions
    mean = np.mean(joint_positions, axis=0)
    standard_deviation = np.std(joint_positions, axis=0)
    distance_from_mean = abs(joint_positions - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    not_outlier_indices = np.where(np.all(not_outlier, axis=1))
    joint_positions_no_outliers = joint_positions[not_outlier_indices]

    init = joint_positions[0]
    fin = joint_positions[-1]

    print(fin)

    joint_positions = np.vstack((init, joint_positions_no_outliers))
    joint_positions = np.vstack((joint_positions, fin))

    # debugging

    greens = np.linspace(255, 0, num=len(joint_positions)) / 255
    reds = np.linspace(0, 255, num=len(joint_positions)) / 255
    blues = np.zeros((len(joint_positions),))
    colors = np.column_stack((reds, greens, blues))
    plt.scatter(np.linspace(1, len(joint_positions[:, 0]), num=len(joint_positions[:, 0])), joint_positions[:, 0],
                c=colors)

    greens = np.zeros((len(joint_positions),))
    blues = np.ones((len(joint_positions),))
    colors = np.column_stack((reds, greens, blues))
    plt.scatter(np.linspace(1, len(joint_positions[:, 1]), num=len(joint_positions[:, 1])), joint_positions[:, 1],
                c=colors)
    # plt.show()

    greens = np.linspace(255, 0, num=len(xy_positions)) / 255
    reds = np.linspace(0, 255, num=len(xy_positions)) / 255
    blues = np.zeros((len(xy_positions),))
    colors = np.column_stack((reds, greens, blues))

    plt.scatter(xy_positions[:, 0], xy_positions[:, 1], c=colors)
    plt.xlim([-0.653, 1.2])
    plt.ylim([-.5, 1.2])
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    # plt.show()

    # convert numpy array to dictionary
    d = dict(enumerate(joint_positions.tolist(), 1))

    # convert dictionary to json
    j = json.dumps(d)

    return j


if __name__ == "__main__":
    main(np.fromstring(sys.argv[1], dtype=float, sep=','), np.fromstring(sys.argv[2], dtype=float, sep=','))
