cartpole_config = {
    'ctrl_freq': 50,
    'pyb_freq': 50,
    'episode_len_sec': 5,
    'normalized_rl_action_space': True,

    # task
    'task': 'stabilization',
    'task_info': {
        'stabilization_goal': [0],
        'stabilization_goal_tolerance': 0.005,
    },
    'cost': 'rl_reward',

    # init
    'randomized_init': True,
    'init_state_randomization_info': {
        'init_x': {
            'distrib': 'uniform',
            'low': -0.5,
            'high': 0.5
        },
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.1,
            'high': 0.1
        },
    },

    # constraint
    'constraints': [{
        'constraint_form': 'bounded_constraint',
        'constrained_variable': 'state',
        'active_dims': [2, 3],
        'lower_bounds': [-0.2, -0.2],
        'upper_bounds': [0.2, 0.2],
    }],
    'done_on_violation': False,

    # custom
    'rew_state_weight': [1, 0, 0, 0],
    'rew_act_weight': 0.001,
    'rew_exponential': True,
    'done_on_out_of_bound': False,
}

cartpole_random_config = {
    **cartpole_config,
    'init_state_randomization_info': {
        'init_x': {
            'distrib': 'uniform',
            'low': -0.5,
            'high': 0.5
        },
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.3,
            'high': 0.3
        },
        'init_theta_dot': {
            'distrib': 'uniform',
            'low': -0.3,
            'high': 0.3
        },
    },
}

quadrotor_config = {
    'ctrl_freq': 50,
    'pyb_freq': 50,
    'episode_len_sec': 5,
    'normalized_rl_action_space': True,

    # task
    'task': 'traj_tracking',
    'task_info': {
        'trajectory_type': 'circle',
        'num_cycles': 1,
        'trajectory_plane': 'xz',
        'trajectory_position_offset': [0, 1],
        'trajectory_scale': 0.7,
    },

    # init
    'randomized_init': True,
    'init_state_randomization_info': {
        'init_x': {
            'distrib': 'uniform',
            'low': -0.4,
            'high': 0.4
        },
        'init_x_dot': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_z': {
            'distrib': 'uniform',
            'low': 0.6,
            'high': 1.4
        },
        'init_z_dot': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.3,
            'high': 0.3
        },
        'init_theta_dot': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
    },

    # constraint
    'constraints': [
        {
            'constraint_form': 'bounded_constraint',
            'constrained_variable': 'state',
            'active_dims': [0, 2],
            'lower_bounds': [-0.5, 0.5],
            'upper_bounds': [0.5, 1.5],
        }
    ],
    'done_on_violation': False,

    # custom
    'quad_type': 2,
    'obs_goal_horizon': 1,
    'rew_exponential': True,
    'done_on_out_of_bound': True,
}

quadrotor_random_config = {
    **quadrotor_config,
    'init_state_randomization_info': {
        'init_x': {
            'distrib': 'uniform',
            'low': -0.7,
            'high': 0.7
        },
        'init_x_dot': {
            'distrib': 'uniform',
            'low': -0.5,
            'high': 0.5
        },
        'init_z': {
            'distrib': 'uniform',
            'low': 0.3,
            'high': 1.7
        },
        'init_z_dot': {
            'distrib': 'uniform',
            'low': -0.5,
            'high': 0.5
        },
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.3,
            'high': 0.3
        },
        'init_theta_dot': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
    },
}