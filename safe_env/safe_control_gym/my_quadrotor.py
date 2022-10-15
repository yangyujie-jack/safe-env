import jax.numpy as jnp
import numpy as np
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor

from safe_env.base import BarrierEnv


class MyQuadrotor(Quadrotor, BarrierEnv):
    def step(self, action):
        feasibility_info = self.get_feasibility_info()
        obs, rew, done, info = super(MyQuadrotor, self).step(action)
        info.update({
            'cost': info['constraint_violation'],
            **feasibility_info,
        })
        return obs, rew, done, info

    def get_feasibility_info(self):
        x, z = self.state[0], self.state[2]
        feasible = abs(x) < 0.1 and abs(z - 1) < 0.1
        infeasible = abs(x) > 0.5 or abs(z - 1) > 0.5
        return {'feasible': feasible, 'infeasible': infeasible}

    @property
    def barrier_input_dim(self):
        return 6

    @staticmethod
    def preprocess(obs):
        return obs[..., :6]

    @staticmethod
    def handcraft_barrier(obs):
        xs, x_dot, zs, z_dot = obs[..., 0], obs[..., 1], obs[..., 2], obs[..., 3]
        bx1 = -0.5 - xs - (xs > -0.5) * 0.2 * x_dot
        bx2 = xs - 0.5 + (xs < 0.5) * 0.2 * x_dot
        bz1 = 0.5 - zs - (zs > 0.5) * 0.2 * z_dot
        bz2 = zs - 1.5 + (zs < 1.5) * 0.2 * z_dot
        barrier = jnp.maximum(jnp.maximum(bx1, bx2), jnp.maximum(bz1, bz2))
        # barrier = (((xs + 0.1 * x_dot) / (0.5 - 0.1 * jnp.minimum(jnp.abs(x_dot), 4))) ** 6 +
        #            ((zs - 1 + 0.1 * z_dot) / (0.5 - 0.1 * jnp.minimum(jnp.abs(z_dot), 4))) ** 6) ** (1 / 6) - 1
        return barrier

    def plot_map(self, ax, x_dot=0.5, z_dot=0):
        from matplotlib.patches import Rectangle

        xs = np.linspace(-0.7, 0.7, 101, dtype=np.float32)
        zs = np.linspace(0.3, 1.7, 101, dtype=np.float32)
        xs, zs = np.meshgrid(xs, zs)

        obs = np.zeros((*xs.shape, 6), dtype=np.float32)
        obs[..., 0] = xs
        obs[..., 1] = x_dot
        obs[..., 2] = zs
        obs[..., 3] = z_dot

        rect = Rectangle((-0.5, 0.5), 1.0, 1.0, fill=False, linestyle='--', color='k')
        ax.add_patch(rect)

        barrier = self.handcraft_barrier(obs)

        return {
            'xs': xs,
            'ys': zs,
            'obs': obs,
            'y_true': None,
            'cbf': barrier,
            'x_label': 'x',
            'y_label': 'z',
        }
