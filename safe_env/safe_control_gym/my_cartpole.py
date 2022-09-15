import numpy as np
from safe_control_gym.envs.gym_control.cartpole import CartPole


class MyCartPole(CartPole):
    def step(self, action):
        feasible = self.goal_reached()
        infeasible = self.constraint_violated()
        barrier = self.get_barrier()
        obs, reward, done, info = super(MyCartPole, self).step(action)
        info.update({
            'cost': info['constraint_violation'],
            'feasible': feasible,
            'infeasible': infeasible,
            'barrier': barrier,
            'next_barrier': self.get_barrier(),
        })
        return obs, reward, done, info

    def goal_reached(self):
        return bool(np.linalg.norm(self.state - self.X_GOAL) <
                    self.TASK_INFO["stabilization_goal_tolerance"])

    def constraint_violated(self):
        c_value = self.constraints.get_values(self)
        return self.constraints.is_violated(self, c_value=c_value)

    def get_barrier(self):
        theta, theta_dot = self.state[2], self.state[3]
        theta_max, theta_dot_max = 0.2, 0.2
        cbf = 0.5 * (-1 + theta ** 2 / theta_max ** 2 + theta_dot ** 2 / theta_dot_max ** 2)
        return cbf

    def plot_map(self, ax):
        from matplotlib.patches import Rectangle

        theta = np.linspace(-0.3, 0.3, 101, dtype=np.float32)
        theta_dot = np.linspace(-0.3, 0.3, 101, dtype=np.float32)
        theta, theta_dot = np.meshgrid(theta, theta_dot)

        obs = np.zeros((*theta.shape, 4), dtype=np.float32)
        obs[..., 0] = 0
        obs[..., 1] = 0
        obs[..., 2] = theta
        obs[..., 3] = theta_dot

        rect = Rectangle((-0.2, -0.2), 0.4, 0.4, fill=False, color='k')
        ax.add_patch(rect)

        theta_max, theta_dot_max = 0.2, 0.2
        cbf = 0.5 * (-1 + theta ** 2 / theta_max ** 2 + theta_dot ** 2 / theta_dot_max ** 2)

        return {
            'xs': theta,
            'ys': theta_dot,
            'obs': obs,
            'y_true': None,
            'handcraft_cbf': cbf,
            'x_label': 'theta [rad]',
            'y_label': 'theta_dot [rad/s]',
        }
