from typing import Tuple

import gym
import numpy as np

from safe_env.base import BarrierEnv


class DoubleIntegrator(BarrierEnv):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.dt = 0.1
        self.state = None

    def reset(self) -> np.ndarray:
        self.state = np.random.uniform(low=-5, high=5, size=2)
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        feasibility_info = self._get_feasibility_info()
        a = np.clip(action, self.action_space.low, self.action_space.high)[0]
        x1, x2 = self.state
        new_x1 = x1 + x2 * self.dt
        new_x2 = x2 + a * self.dt
        self.state[0] = new_x1
        self.state[1] = new_x2
        done = feasibility_info['feasible'] or feasibility_info['infeasible']
        return self._get_obs(), 0.0, done, feasibility_info

    def _get_obs(self):
        return np.copy(self.state)

    def _get_feasibility_info(self):
        x1, x2 = self.state
        feasible = abs(x1) < 0.1 and abs(x2) < 0.1
        infeasible = abs(x1) > 5 or abs(x2) > 5
        return {'feasible': feasible, 'infeasible': infeasible}

    def plot_map(self, ax):
        from matplotlib.patches import Rectangle

        x1 = np.linspace(-5, 5, 101)
        x2 = np.linspace(-5, 5, 101)
        x1_grid, x2_grid = np.meshgrid(x1, x2)
        obs = np.stack((x1_grid, x2_grid), axis=2)

        x2_min = -np.sqrt(2 * (x1 + 5))
        x2_max = np.sqrt(2 * (5 - x1))
        ax.plot(x1, x2_min, color='k')
        ax.plot(x1, x2_max, color='k')
        rect = Rectangle((-0.5, -0.5), 1, 1, fill=False, linestyle='--', color='k')
        ax.add_patch(rect)

        feasible = (x2_grid >= x2_min) & (x2_grid <= x2_max)
        y_true = feasible * 0 + ~feasible * 1

        barrier = (x2_grid >= 0) * (x1_grid - 5 + x2_grid) + (x2_grid <= 0) * (-5 - x1_grid - x2_grid)

        return {
            'xs': x1_grid,
            'ys': x2_grid,
            'obs': obs,
            'y_true': y_true,
            'handcraft_barrier': barrier,
            'x_label': 'x [m]',
            'y_label': 'v [m/s]',
        }
