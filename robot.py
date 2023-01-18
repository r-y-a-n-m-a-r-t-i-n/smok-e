import numpy as np


class Robot:
    def __init__(self):
        # Constants

        self.L_1 = 1  # link 1 length
        self.L_2 = 1  # link 2 length
        self.m_1 = 1  # link 1 mass
        self.m_2 = 1  # link 2 mass
        self.g = 9.81  # gravity
        self.h = .33    # distance of arm base from ground
        # self.g = 0.0  # zero-g for debugging

        self.q = np.array(
            [np.pi/2.0,  # theta_r (rotate angle)
             np.pi / 4.0,  # theta_s (shoulder angle)
             np.pi / 4.0,  # theta_e (elbow angle)
             ]
        )

        # Initial state [q, q_dot]
        self.x_0 = np.array(
            [
                0.0,  # theta_r (rotate angle)
                np.pi / 4.0,  # theta_s (shoulder angle)
                np.pi / 4.0,  # theta_e (elbow angle)

                0.0,  # theta_dot_r
                0.0,  # theta_dot_s (shoulder joint velocity)
                0.0,  # theta_dot_e (elbow joint velocity)

                # deal with these later
                # 2 * np.pi / 3,  # theta_2 (gripper open/close)
                # np.pi / 2,  # theta_3 (gripper rotation)
            ]
        )
        # State
        self.x = self.x_0

    def f(self, x, u, L_1=None):
        """
        x_dot = f(x, u)
        x_dot = [q_dot, q_ddot]

        qddot = B(q).inv @ (-C(qdot, q) - G(q)) + F
        """

        B = np.array([
            [
                ((
                         self.m_1 + self.m_2) * self.L_1 ** 2 + self.m_2 * self.L_2 ** 2 + 2 * self.m_2 * self.L_1 * self.L_2 * np.cos(
                    x[1])),

                (self.m_2 * self.L_2 ** 2 - self.m_2 * self.L_1 * self.L_2 * np.cos(x[1])),
            ],
            [
                (self.m_2 * self.L_2 ** 2 + self.m_2 * self.L_1 * self.L_2 * np.cos(x[1])),

                self.m_2 * self.L_2,
            ]
        ])

        C = np.array([
            (-self.m_2 * self.L_1 * self.L_2 * np.sin(x[1]) * (2 * x[2] * x[3] + x[3] ** 2)),

            (-self.m_2 * self.L_1 * self.L_2 * np.sin(x[1]) * x[2] * x[3]),
        ])

        G = np.array([
            (-(self.m_1 + self.m_2) * self.L_1 * self.g * np.sin(x[0]) - self.m_2 * self.L_2 * self.g * np.sin(
                x[0] + x[1])),

            (-self.m_2 * self.L_2 * self.g * np.sin(x[0] + x[1])),

        ])

        q_ddot = np.linalg.inv(B) @ (-1 * C - G) + u

        theta_0_ddot = q_ddot[0]

        theta_1_ddot = q_ddot[1]

        return np.array([x[2], x[3], theta_0_ddot, theta_1_ddot])

    def r_q(self, q):
        # origin is at arm base (so ground z = -h)
        # y is robot forward, x is robot right, z is up

        theta_r = q[0]
        theta_s = q[1]
        theta_e = q[2]

        L_1 = self.L_1
        L_2 = self.L_2

        r_x = (L_1 * np.cos(theta_s) + L_2 * np.cos(theta_s + theta_e)) * np.sin(theta_r)
        r_y = (L_1 * np.cos(theta_s) + L_2 * np.cos(theta_s + theta_e)) * np.cos(theta_r)
        r_z = L_1 * np.sin(theta_s) + L_2 * np.sin(theta_s + theta_e)

        return np.array((r_x, r_y, r_z))
