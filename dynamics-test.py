import numpy as np
import matplotlib.pyplot as plt


class Robot:
    def __init__(self):
        # Constants

        self.L_0 = 1  # link 0 length
        self.L_1 = 1  # link 1 length
        self.m_0 = 1  # link 0 mass
        self.m_1 = 1  # link 1 mass
        self.g = 9.81  # gravity

        # Initial state [q, q_dot]
        self.x_0 = np.array(
            [
                np.pi / 4.0,  # theta_0 (shoulder angle)
                np.pi / 4.0,  # theta_1 (elbow angle)
                0.0,  # theta_dot_0 (shoulder joint velocity)
                0.0,  # theta_dot_1 (elbow joint velocity)

                # deal with these later
                # 2 * np.pi / 3,  # theta_2 (gripper open/close)
                # np.pi / 2,  # theta_3 (gripper rotation)
            ]
        )
        # State
        self.x = self.x_0

    def f(self, x, u, l_0=None):
        """
        x_dot = f(x, u)
        x_dot = [q_dot, q_ddot]

        qddot = B(q).inv @ (-C(qdot, q) - G(q)) + F
        """

        B = np.array([
            [
                ((
                         self.m_0 + self.m_1) * self.L_0 ** 2 + self.m_1 * self.L_1 ** 2 + 2 * self.m_1 * self.L_0 * self.L_1 * np.cos(
                    x[1])),

                (self.m_1 * self.L_1 ** 2 - self.m_1 * self.L_0 * self.L_1 * np.cos(x[1])),
            ],
            [
                (self.m_1 * self.L_1 ** 2 + self.m_1 * self.L_0 * self.L_1 * np.cos(x[1])),

                self.m_1 * self.L_1,
            ]
        ])

        C = np.array([
            (-self.m_1 * self.L_0 * self.L_1 * np.sin(x[1]) * (2 * x[2] * x[3] + x[3] ** 2)),

            (-self.m_1 * self.L_0 * self.L_1 * np.sin(x[1]) * x[2] * x[3]),
        ])

        G = np.array([
            (-(self.m_0 + self.m_1) * self.L_0 * self.g * np.sin(x[0]) - self.m_1 * self.L_1 * self.g * np.sin(
                x[0] + x[1])),

            (-self.m_1 * self.L_1 * self.g * np.sin(x[0] + x[1])),

        ])

        q_ddot = np.linalg.inv(B) @ (-1 * C - G) + u

        theta_0_ddot = q_ddot[0]

        theta_1_ddot = q_ddot[1]

        return np.array([x[2], x[3], theta_0_ddot, theta_1_ddot])

    def step(self, x, u, dt):
        self.x += self.f(x, u) * dt


if __name__ == "__main__":
    # Instantiate robot
    robot = Robot()

    # Simulation parameters
    dt = 0.0001
    t = 0.0
    t_max = 10.0

    # Graphing parameters
    robot_state_history = np.reshape(robot.x_0, (4, 1))
    t_history = [0.0]

    # Run simulation
    while t < t_max:
        # u = [tau_0, tau_1]

        u = np.array([5.0, -5.0])

        robot.step(x=robot.x, u=u, dt=dt)
        robot_state_history = np.hstack(
            (robot_state_history, np.reshape(robot.x, (4, 1)))
        )
        t += dt
        t_history.append(t)

    # Plot

    x_0 = robot.L_0 * np.cos(robot_state_history[0, :])
    y_0 = robot.L_0 * np.sin(robot_state_history[0, :])
    x_1 = robot.L_0 * np.cos(robot_state_history[0, :]) + robot.L_1 * np.cos(robot_state_history[0, :] + robot_state_history[1, :])
    y_1 = robot.L_0 * np.sin(robot_state_history[0, :]) + robot.L_1 * np.sin(robot_state_history[0, :] + robot_state_history[1, :])

    plt.figure()
    plt.plot(
        -robot_state_history[0, :] * np.sin(robot_state_history[1, :]),
        robot_state_history[0, :] * np.cos(robot_state_history[1, :]),
    )
    plt.xlabel("x-position")
    plt.ylabel("z-position")
    plt.legend()
    plt.show()
