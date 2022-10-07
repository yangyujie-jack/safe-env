import numpy as np
from safe_control_gym.envs.gym_control.cartpole import CartPole

from safe_env.base import BarrierEnv


class MyCartPole(CartPole, BarrierEnv):
    def step(self, action):
        feasible = self.goal_reached()
        infeasible = self.constraint_violated()
        obs, reward, done, info = super(MyCartPole, self).step(action)
        info.update({
            'cost': info['constraint_violation'],
            'feasible': feasible,
            'infeasible': infeasible,
        })
        return obs, reward, done, info

    def goal_reached(self):
        return bool(np.linalg.norm(self.state - self.X_GOAL) <
                    self.TASK_INFO['stabilization_goal_tolerance'])

    def constraint_violated(self):
        c_value = self.constraints.get_values(self)
        return self.constraints.is_violated(self, c_value=c_value)

    @staticmethod
    def handcraft_barrier(obs):
        theta, theta_dot = obs[..., 2], obs[..., 3]
        theta_max, theta_dot_max = 0.2, 0.2
        barrier = 0.5 * (-1 + theta ** 2 / theta_max ** 2 + theta_dot ** 2 / theta_dot_max ** 2)
        return barrier

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

        barrier = self.handcraft_barrier(obs)

        return {
            'xs': theta,
            'ys': theta_dot,
            'obs': obs,
            'y_true': None,
            'handcraft_barrier': barrier,
            'x_label': r'$\theta$',
            'y_label': r'$\dot{\theta}$',
        }
