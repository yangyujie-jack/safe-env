import numpy as np


def normalize_obs(pos, robot_pos, robot_mat):
    vec = (pos - robot_pos) @ robot_mat
    x, y = vec[..., 0], vec[..., 1]
    z = x + 1j * y
    dist = np.abs(z)
    angle = np.angle(z)
    return np.stack((np.exp(-dist), np.cos(angle), np.sin(angle)), axis=-1)


def relative_dist_angle(pos: np.ndarray, center: np.ndarray, rot_mat: np.ndarray, angle_normalize: bool = False):
    """
    pos: (N, 2)
    center: (B, 2)
    rot_mat: (B, 2, 2)
    returns: (B, N, 3)
    """
    rel = pos - center[:, np.newaxis, :]
    vec = rel @ rot_mat  # (B, N, 2)
    x, y = vec[..., 0], vec[..., 1]  # (B, N)
    z = x + 1j * y  # (B, N)
    dist = np.abs(z)  # (B, N)
    angle = np.angle(z)  # (B, N)
    if angle_normalize:
        angle %= (2 * np.pi)
    return dist, angle


def lidar_max(sensor: np.ndarray, bin: np.ndarray, lidar_num_bins: int):
    """
    sensor: (B, N)
    bin: (B, N)
    returns: (B, lidar_num_bins)
    Not sure if there is a better way to do this.
    """
    mask = bin[:, :, np.newaxis] == np.arange(lidar_num_bins)  # (B, N, lidar_num_bins)
    sensor = np.broadcast_to(sensor[:, :, np.newaxis], mask.shape)
    return np.max(sensor, axis=1, where=mask, initial=0.0)


def obs_lidar_pseudo(
        pos: np.ndarray,
        center: np.ndarray,
        rot_mat: np.ndarray,
        lidar_num_bins: int,
        lidar_max_dist: float,
        lidar_exp_gain: float,
        lidar_alias: bool,
):
    """
    pos: (N, 2)
    center: (B, 2)
    rot_mat: (B, 2, 2)
    returns: (B, lidar_num_bins)
    """
    dist, angle = relative_dist_angle(pos, center, rot_mat, True)

    bin_size = (np.pi * 2) / lidar_num_bins
    bin = (angle / bin_size).astype(np.int64)  # (B, N), truncated towards 0
    bin_angle = bin * bin_size
    if lidar_max_dist is None:
        sensor = np.exp(-lidar_exp_gain * dist)
    else:
        sensor = np.maximum(lidar_max_dist - dist, 0.0) / lidar_max_dist
    lidar = lidar_max(sensor, bin, lidar_num_bins)
    if lidar_alias:
        alias = (angle - bin_angle) / bin_size
        assert np.all((alias >= 0) & (alias <= 1)), f'bad alias {alias}, dist {dist}, angle {angle}, bin {bin}'
        bin_plus = (bin + 1) % lidar_num_bins
        bin_minus = (bin - 1) % lidar_num_bins
        sensor_plus = alias * sensor
        sensor_minus = (1 - alias) * sensor
        lidar = np.maximum(lidar, lidar_max(sensor_plus, bin_plus, lidar_num_bins))
        lidar = np.maximum(lidar, lidar_max(sensor_minus, bin_minus, lidar_num_bins))
    return lidar


def obs_lidar_pseudo2(
        pos: np.ndarray,
        center: np.ndarray,
        rot_mat: np.ndarray,
        lidar_num_bins: int,
        hazards_size: float,
):
    """
    pos: (N, 2)
    center: (B, 2)
    rot_mat: (B, 2, 2)
    returns: (B, lidar_num_bins)
    """
    dist, angle = relative_dist_angle(pos, center, rot_mat, True)
    dist = dist[..., np.newaxis]
    dist_mask = dist >= hazards_size
    angle = angle[..., np.newaxis]
    bin_angle = np.linspace(0, 2 * np.pi, num=lidar_num_bins, endpoint=False)
    bin_angle = bin_angle[np.newaxis, np.newaxis, :]
    delta_angle = (angle - bin_angle + np.pi) % (2 * np.pi) - np.pi
    angle_mask = np.abs(delta_angle) <= np.pi / 2
    dist_to_bin = dist * np.abs(np.sin(delta_angle))
    bin_mask = dist_to_bin <= hazards_size
    dist_to_bin = np.clip(dist_to_bin, None, hazards_size)
    a = dist * np.abs(np.cos(delta_angle))
    b = np.sqrt(hazards_size ** 2 - dist_to_bin ** 2)
    obs_dist = dist_mask * (a - b) - ~dist_mask * (angle_mask * (a + b) + ~angle_mask * (b - a))
    lidar = dist_mask * angle_mask * bin_mask * np.exp(-obs_dist) + ~dist_mask * np.exp(-obs_dist)
    lidar = np.max(lidar, axis=1)
    return lidar


def generate_obs(config):
    """
    config keys:
        goal_pos,
        robot_pos,
        robot_rot,
        robot_vel,
        hazards_pos,
    """
    obs = {}

    batch_size = config['robot_pos'].shape[0]
    rot_cos, rot_sin = np.cos(config['robot_rot']), np.sin(config['robot_rot'])
    robot_mat = np.array([[rot_cos, -rot_sin], [rot_sin, rot_cos]], dtype=np.float32)

    robot_vel = np.array(config['robot_vel'], dtype=np.float32)
    obs['velocimeter'] = np.broadcast_to(robot_vel @ robot_mat, (batch_size, 2))

    obs['hazards_lidar'] = obs_lidar_pseudo2(
        pos=config['hazards_pos'],
        center=config['robot_pos'],
        rot_mat=robot_mat,
        lidar_num_bins=config['lidar_num_bins'],
        hazards_size=config['hazards_size']
    )

    obs['goal_pos'] = normalize_obs(config['goal_pos'], config['robot_pos'], robot_mat)

    flat_obs = []
    for v in obs.values():
        flat_obs.append(v)
    flat_obs = np.concatenate(flat_obs, axis=1)

    return flat_obs


if __name__ == '__main__':
    # B = 10
    # center = np.zeros((B, 2))
    # rot_mat = np.stack([np.eye(2)] * B)
    # pos = np.random.uniform(-2, 2, size=(4, 2))
    pos = np.array([[0, 0]])
    center = np.array([[0, 0]])
    rot_mat = np.eye(2)
    lidar = obs_lidar_pseudo2(pos, center, rot_mat, lidar_num_bins=10, hazards_size=1)
    print(lidar)
