point_goal_config = {
    'robot_base': 'xmls/point.xml',

    'lidar_num_bins': 36,

    'task': 'goal',

    'goal_keepout': 0.4,
    'goal_size': 0.3,

    'constrain_hazards': True,

    'hazards_num': 8,
    'hazards_keepout': 0.3,
    'hazards_size': 0.2,
}

car_goal_config = {
    **point_goal_config,
    'robot_base': 'xmls/car.xml',
}
