import gym

from safe_env.safe_control_gym.config import cartpole_config, quadrotor_config, \
    cartpole_random_config, quadrotor_random_config
from safe_env.safe_control_gym.my_cartpole import MyCartPole
from safe_env.safe_control_gym.my_quadrotor import MyQuadrotor
from safe_env.safety_gym.config import point_goal_config, car_goal_config
from safe_env.safety_gym.my_engine import MyEngine
from safe_env.simple_task.double_integrator import DoubleIntegrator
from safe_env.simple_task.point_robot import PointRobot


def register():
    # simple tasks
    gym.register(
        id='DoubleIntegrator-v0',
        entry_point=DoubleIntegrator,
        max_episode_steps=2
    )

    gym.register(
        id='PointRobot-v0',
        entry_point=PointRobot,
        max_episode_steps=10
    )

    # safe control gym
    gym.register(
        id='CartPole-v0',
        entry_point=MyCartPole,
        kwargs=cartpole_config
    )

    gym.register(
        id='CartPole-v1',
        entry_point=MyCartPole,
        kwargs=cartpole_random_config,
        max_episode_steps=50
    )

    gym.register(
        id='Quadrotor-v0',
        entry_point=MyQuadrotor,
        kwargs=quadrotor_config
    )

    gym.register(
        id='Quadrotor-v1',
        entry_point=MyQuadrotor,
        kwargs=quadrotor_random_config,
        max_episode_steps=50
    )

    # safety gym
    gym.register(
        id='PointGoal-v0',
        entry_point=MyEngine,
        kwargs={'config': point_goal_config}
    )

    gym.register(
        id='CarGoal-v0',
        entry_point=MyEngine,
        kwargs={'config': car_goal_config}
    )
