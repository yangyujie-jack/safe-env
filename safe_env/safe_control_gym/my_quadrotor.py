import gym
import numpy as np
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor


class MyQuadrotor(Quadrotor):
    def _set_observation_space(self):
        super(MyQuadrotor, self)._set_observation_space()
        self.observation_space = gym.spaces.Box(
            low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max,
            shape=self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        feasibility_info = self.get_feasibility_info()
        barrier = self.get_barrier()
        obs, rew, done, info = super(MyQuadrotor, self).step(action)
        next_barrier = self.get_barrier()
        info.update({
            'cost': info['constraint_violation'],
            **feasibility_info,
            'barrier': barrier,
            'next_barrier': next_barrier,
        })
        return obs, rew, done, info

    def get_feasibility_info(self):
        x, z = self.state[0], self.state[2]
        feasible = abs(x) < 0.1 and abs(z - 1) < 0.1
        infeasible = abs(x) > 0.5 or abs(z - 1) > 0.5
        return {'feasible': feasible, 'infeasible': infeasible}

    def get_barrier(self):
        xs, x_dot, zs, z_dot = self.state[:4]
        bx1 = -0.5 - xs - (xs > -0.5) * 0.2 * x_dot
        bx2 = xs - 0.5 + (xs < 0.5) * 0.2 * x_dot
        bz1 = 0.5 - zs - (zs > 0.5) * 0.2 * z_dot
        bz2 = zs - 1.5 + (zs < 1.5) * 0.2 * z_dot
        cbf = np.maximum(np.maximum(bx1, bx2), np.maximum(bz1, bz2))
        return cbf

    def plot_map(self, ax, x_dot=0.5, z_dot=0, theta=0, theta_dot=0):
        from matplotlib.patches import Rectangle

        xs = np.linspace(-0.7, 0.7, 101, dtype=np.float32)
        zs = np.linspace(0.3, 1.7, 101, dtype=np.float32)
        xs, zs = np.meshgrid(xs, zs)

        obs = np.zeros((*xs.shape, 6), dtype=np.float32)
        obs[..., 0] = xs
        obs[..., 1] = x_dot
        obs[..., 2] = zs
        obs[..., 3] = z_dot
        obs[..., 4] = theta
        obs[..., 5] = theta_dot

        rect = Rectangle((-0.5, 0.5), 1.0, 1.0, fill=False, color='k')
        ax.add_patch(rect)
        rect = Rectangle((-0.1, 0.9), 0.2, 0.2, fill=False, color='k', linestyle='--')
        ax.add_patch(rect)

        bx1 = -0.5 - xs - (xs > -0.5) * 0.2 * x_dot
        bx2 = xs - 0.5 + (xs < 0.5) * 0.2 * x_dot
        bz1 = 0.5 - zs - (zs > 0.5) * 0.2 * z_dot
        bz2 = zs - 1.5 + (zs < 1.5) * 0.2 * z_dot
        cbf = np.maximum(np.maximum(bx1, bx2), np.maximum(bz1, bz2))

        return {
            'xs': xs,
            'ys': zs,
            'obs': obs,
            'y_true': None,
            'handcraft_cbf': cbf,
            'x_label': 'x [m]',
            'y_label': 'z [m]',
        }
