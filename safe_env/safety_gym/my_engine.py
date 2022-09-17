import gym
import jax.numpy as jnp
import numpy as np
from safety_gym.envs.engine import Engine

from safe_env.base import BarrierEnv
from safe_env.safety_gym.generate_observations import normalize_obs, obs_lidar_pseudo2


class MyEngine(Engine, BarrierEnv):
    def build_observation_space(self):
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.lidar_num_bins + 5,), dtype=np.float32)

    def obs(self):
        self.sim.forward()
        robot_pos = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        velocimeter = self.world.get_sensor('velocimeter')[:2]
        hazards_lidar = self.obs_lidar(self.hazards_pos, None)
        goal_obs = normalize_obs(self.goal_pos[np.newaxis, :], robot_pos, robot_mat)[0]
        return np.concatenate((velocimeter, hazards_lidar, goal_obs)).astype(np.float32)

    def step(self, action):
        feasibility_info = self.get_feasibility_info()
        obs, reward, done, info = super(MyEngine, self).step(action)
        info.update({
            **feasibility_info,
        })
        return obs, reward, done, info

    def obs_lidar_pseudo(self, positions):
        return obs_lidar_pseudo2(
            pos=np.array(positions)[:, :2],
            center=self.robot_pos[np.newaxis, :2],
            rot_mat=self.world.robot_mat()[np.newaxis, :2, :2],
            lidar_num_bins=self.config['lidar_num_bins'],
            hazards_size=self.hazards_size
        )[0]

    def get_feasibility_info(self):
        robot_pos = self.robot_pos[np.newaxis, :2]
        hazards_pos = np.array(self.hazards_pos)[:, :2]
        hazards_dist = np.linalg.norm(hazards_pos - robot_pos, axis=1)
        feasible = self.goal_met()
        infeasible = np.min(hazards_dist) <= self.hazards_size
        return {'feasible': feasible, 'infeasible': infeasible}

    @staticmethod
    def handcraft_barrier(obs):
        lidar_num_bins = 36
        rel_vel = obs[..., :2]
        hazards_lidar = jnp.clip(obs[..., 2:2 + lidar_num_bins], a_min=1e-4)
        bin_dist = -jnp.log(hazards_lidar)
        bin_angle = jnp.linspace(0, 2 * np.pi, lidar_num_bins + 1)[:-1]
        bin_proj_vec = jnp.stack((jnp.cos(bin_angle), jnp.sin(bin_angle)))
        bin_dist_dot = -jnp.matmul(rel_vel[..., jnp.newaxis, :], bin_proj_vec)[..., 0, :]
        barrier = 0.1 - bin_dist - 0.1 * bin_dist_dot
        barrier = jnp.max(barrier, axis=-1)
        return barrier

    def plot_map(self, ax, robot_vel=(1, 0)):
        from matplotlib.patches import Circle
        from safe_env.safety_gym.generate_observations import generate_obs

        config = {
            **self.config,
            'robot_rot': 0,
            '_seed': 0,
        }
        env = MyEngine(config)
        env.reset()
        goal_pos = env.goal_pos[:2]
        hazards_pos = np.stack(env.hazards_pos, axis=0)[:, :2]

        for pos in hazards_pos:
            circle = Circle(pos, self.hazards_size, fill=False, linestyle='--', color='k')
            ax.add_patch(circle)
        circle = Circle(goal_pos, self.goal_size, fill=False, color='k')
        ax.add_patch(circle)

        n = 101
        x_lim = (-2, 2)
        y_lim = (-2, 2)
        xs = np.linspace(x_lim[0], x_lim[1], n)
        ys = np.linspace(y_lim[0], y_lim[1], n)
        xs, ys = np.meshgrid(xs, ys)
        robot_pos = np.stack((xs, ys), axis=2).reshape(-1, 2)
        obs = generate_obs({
            **config,
            'goal_pos': goal_pos,
            'robot_pos': robot_pos,
            'robot_vel': robot_vel,
            'hazards_pos': hazards_pos,
        }).reshape(n, n, -1)

        barrier = self.handcraft_barrier(obs)

        return {
            'xs': xs,
            'ys': ys,
            'obs': obs,
            'y_true': None,
            'handcraft_barrier': barrier,
            'x_label': 'x [m]',
            'y_label': 'y [m]',
        }
