import numpy as np
import matplotlib.pyplot as plt


class Robot:
    def __init__(self):
        # Constants

        self.L_0 = 1  # link 1 length
        self.L_1 = 1  # link 2 length
        self.m_1 = 1  # link 1 mass
        self.m_2 = 1  # link 2 mass
        self.g = 9.81  # gravity

        # x_0 = L_0 * cos(theta_0)
        # y_0 = L_0 * sin(theta_0)
        # x_1 = L_0 * cos(theta_0) + L_1 * cos(theta_0 + theta_1)
        # y_1 = L_0 * sin(theta_0) + L_1 * sin(theta_0 + theta_1)

        # Initial state [q, q_dot]
        self.x_0 = np.array(
            [
                np.pi / 4,  # theta_0 (shoulder angle)
                np.pi / 4,  # theta_1 (elbow angle)
                0,  # theta_dot_0 (shoulder joint velocity)
                0,  # theta_dot_1 (elbow joint velocity)

                # deal with these later
                # 2 * np.pi / 3,  # theta_2 (gripper open/close)
                # np.pi / 2,  # theta_3 (gripper rotation)
            ]
        )
        # State
        self.x = self.x_0

    def f(self, x, u):
        """
        x_dot = f(x, u)
        x_dot = [q_dot, q_ddot]
        """

        theta_0_ddot = 0

        theta_1_ddot = 0

        return [x[2], x[3], theta_0_ddot, theta_1_ddot]

    def step(self, x, u, dt):
        self.x += self.f(x, u) * dt


if __name__ == "__main__":
    # Instantiate robot
    robot = Robot()

    # Simulation parameters
    dt = 0.0001
    t = 0
    t_max = 10

    # Graphing parameters
    robot_state_history = np.reshape(robot.x_0, (4, 1))
    t_history = [0]

    # Run simulation
    while t < t_max:

        # u = [tau_0, tau_1]

        u = None

        robot.step(x=robot.x, u=u, dt=dt)
        robot_state_history = np.hstack(
            (robot_state_history, np.reshape(robot.x, (4, 1)))
        )
        t += dt
        t_history.append(t)

    # Plot
    plt.figure()
    plt.plot(
        -robot_state_history[0, :] * np.sin(robot_state_history[1, :]),
        robot_state_history[0, :] * np.cos(robot_state_history[1, :]),
    )
    plt.xlabel("x-position")
    plt.ylabel("z-position")
    plt.legend()
    plt.show()
