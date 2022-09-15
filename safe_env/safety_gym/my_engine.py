import gym
import numpy as np
from safety_gym.envs.engine import Engine

from safety_certificate.env.generate_observations import normalize_obs, obs_lidar_pseudo2


class MyEngine(Engine):
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
        barrier = self.get_barrier()
        obs, reward, done, info = super(MyEngine, self).step(action)
        next_barrier = self.get_barrier()
        info.update({
            **feasibility_info,
            'barrier': barrier,
            'next_barrier': next_barrier,
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

    def get_barrier(self):
        robot_pos = self.robot_pos[np.newaxis, :2]
        robot_vel = self.world.robot_vel()[:2]
        hazards_pos = np.array(self.hazards_pos)[:, :2]
        hazards_vec = hazards_pos - robot_pos
        hazards_dist = np.linalg.norm(hazards_vec, axis=1)
        hazards_angle = np.arctan2(hazards_vec[:, 1], hazards_vec[:, 0])
        dist_dot = -robot_vel[0] * np.cos(hazards_angle) - robot_vel[1] * np.sin(hazards_angle)
        barrier = 0.1 + self.hazards_size ** 2 - hazards_dist ** 2 - 0.1 * dist_dot
        return np.max(barrier)

    def seed(self, seed=None):
        super(MyEngine, self).seed(seed)
        self.action_space.seed(seed)

    def plot_map(self, ax):
        from safety_certificate.env.generate_observations import generate_obs
        from matplotlib.patches import Circle

        config = {
            **self.config,
            'robot_rot': 0,
            '_seed': 0,
        }
        env = MyEngine(config)
        env.reset()
        goal_pos = env.goal_pos[:2]
        hazards_pos = np.stack(env.hazards_pos, axis=0)[:, :2]

        n = 101
        x_lim = (-2, 2)
        y_lim = (-2, 2)
        xs = np.linspace(x_lim[0], x_lim[1], n)
        ys = np.linspace(y_lim[0], y_lim[1], n)
        xs, ys = np.meshgrid(xs, ys)
        robot_pos = np.stack((xs, ys), axis=2).reshape(-1, 2)
        robot_vel = (1, 0)
        obs = generate_obs({
            **config,
            'goal_pos': goal_pos,
            'robot_pos': robot_pos,
            'robot_vel': robot_vel,
            'hazards_pos': hazards_pos,
        }).reshape(n, n, -1)

        lidar_num_bins = config['lidar_num_bins']
        rel_vel = obs[..., :2]
        hazards_lidar = np.clip(obs[..., 2:2 + lidar_num_bins], a_min=1e-4, a_max=None)
        bin_dist = -np.log(hazards_lidar)
        bin_angle = np.linspace(0, 2 * np.pi, num=lidar_num_bins, endpoint=False)
        bin_proj_vec = np.stack((np.cos(bin_angle), np.sin(bin_angle)), axis=0)
        bin_dist_dot = -np.dot(rel_vel, bin_proj_vec)
        cbf = 0.1 - bin_dist - 0.1 * bin_dist_dot
        cbf = np.max(cbf, axis=-1)

        for pos in hazards_pos:
            circle = Circle(pos, self.hazards_size, fill=False, linestyle='--', color='k')
            ax.add_patch(circle)
        circle = Circle(goal_pos, self.goal_size, fill=False, color='k')
        ax.add_patch(circle)

        return {
            'xs': xs,
            'ys': ys,
            'obs': obs,
            'y_true': None,
            'handcraft_cbf': cbf,
            'x_label': 'x [m]',
            'y_label': 'y [m]',
        }


if __name__ == '__main__':
    from safety_certificate.env.config import point_goal_config

    env = MyEngine(point_goal_config)
    env.seed(0)
    np.random.seed(0)
    obs = env.reset()
    # print(obs.shape)

    for i in range(100):
        action = [1, 0]
        obs, rew, done, info = env.step(action)
        print(
            env.robot_pos[:2],
            env.world.get_sensor('velocimeter')[:2]
        )
