from typing import Tuple

import gym
import numpy as np

from safe_env.base import BarrierEnv


class ReachAvoid(BarrierEnv):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.hazard_size = 0.5
        self.dt = 0.1
        self.state = None

    def reset(self) -> np.ndarray:
        self.state = np.random.uniform(low=[-1.5, -1.5, 0.5, np.pi / 4], high=[1.5, 1.5, 1.5, 3 * np.pi / 4])
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        feasibility_info = self._get_feasibility_info()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state = self.state + self._dynamics(self.state, action) * self.dt
        done = feasibility_info['feasible'] or feasibility_info['infeasible']
        return self._get_obs(), 0.0, done, feasibility_info

    @staticmethod
    def _dynamics(s, u):
        v = s[2]
        theta = s[3]

        dot_x = v * np.cos(theta)
        dot_y = v * np.sin(theta)
        dot_v = u[0]
        dot_theta = u[1]

        dot_s = np.array([dot_x, dot_y, dot_v, dot_theta], dtype=np.float32)
        return dot_s

    def _get_obs(self):
        obs = np.zeros(5, dtype=np.float32)
        obs[:3] = self.state[:3]
        theta = self.state[3]
        obs[3] = np.cos(theta)
        obs[4] = np.sin(theta)
        return obs

    def _get_feasibility_info(self):
        feasible = abs(self.state[0]) > 1.5 or abs(self.state[1]) > 1.5
        infeasible = np.linalg.norm(self.state[:2]) <= self.hazard_size
        return {'feasible': feasible, 'infeasible': infeasible}

    def _get_avoidable(self, state):
        x, y, v, theta = state

        hazard_vec = np.array([-x, -y])
        dist = np.linalg.norm(hazard_vec)
        if dist <= self.hazard_size:
            return False

        velocity_vec = np.array([v * np.cos(theta), v * np.sin(theta)])
        velocity = np.linalg.norm(velocity_vec)
        velocity = np.clip(velocity, 1e-6, None)
        cos_theta = np.dot(velocity_vec, hazard_vec) / (velocity * dist)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        delta = self.hazard_size ** 2 - (dist * sin_theta) ** 2
        if cos_theta <= 0 or delta < 0:
            return True

        acc = self.action_space.low[0]
        if np.cross(velocity_vec, hazard_vec) >= 0:
            omega = self.action_space.low[1]
        else:
            omega = self.action_space.high[1]
        action = np.array([acc, omega])
        s = np.copy(state)
        while s[2] > 0:
            s = s + self._dynamics(s, action) * self.dt
            dist = np.linalg.norm([-s[0], -s[1]])
            if dist <= self.hazard_size:
                return False
        return True

    def plot_map(self, ax, v: float = 1.0, theta: float = np.pi / 2):
        from matplotlib.patches import Circle

        n = 101
        xs = np.linspace(-1.5, 1.5, n)
        ys = np.linspace(-1.5, 1.5, n)
        xs, ys = np.meshgrid(xs, ys)
        vs = v * np.ones_like(xs)
        thetas = theta * np.ones_like(xs)
        obs = np.stack((xs, ys, vs, np.cos(thetas), np.sin(thetas)), axis=-1)

        avoidable = np.zeros_like(xs)
        for i in range(n):
            for j in range(n):
                avoidable[i, j] = float(self._get_avoidable([xs[i, j], ys[i, j], v, theta]))
        ax.contour(xs, ys, avoidable - 0.5, levels=[0], colors='k')
        circle = Circle((0.0, 0.0), self.hazard_size, fill=False, linestyle='--', color='k')
        ax.add_patch(circle)

        y_true = 1 - float(avoidable)

        d = np.linalg.norm(obs[..., :2], axis=-1)
        hazard_angle = np.arctan2(-obs[..., 1], -obs[..., 0])
        heading_angle = np.arctan2(obs[..., 4], obs[..., 3])
        d_dot = -obs[..., 2] * np.cos(hazard_angle - heading_angle)
        barrier = 0.5 + self.hazard_size ** 2 - d ** 2 - 0.2 * d_dot

        return {
            'xs': xs,
            'ys': ys,
            'obs': obs,
            'y_true': y_true,
            'handcraft_barrier': barrier,
            'x_label': 'x [m]',
            'y_label': 'y [m]',
        }
