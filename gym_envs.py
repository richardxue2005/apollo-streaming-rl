import gymnasium as gym

import robocasa_robosuite.robosuite.robosuite as suite
from robocasa_robosuite.robosuite.robosuite.wrappers import GymWrapper

from streaming_drl.normalization_wrappers import NormalizeObservation, ScaleReward
from streaming_drl.time_wrapper import AddTimeInfo

def make_lift_env(render=False):
    robosuite_env = suite.make(
        "Lift",
        robots="Panda",
        use_camera_obs=False,
        has_offscreen_renderer=False,
        has_renderer=render,
        reward_shaping=True,
        control_freq=20,
    )
    
    env = GymWrapper(robosuite_env)
    env.spec = gym.envs.registration.EnvSpec(
        id="Lift-Panda-v0",
        entry_point="streaming_rl.gym_envs:make_lift_env",
        max_episode_steps=1000,
        reward_threshold=200.0,
    )

    return env

gym.register(
    id="Lift-Panda-v0",
    entry_point="streaming_rl.gym_envs:make_lift_env",
    max_episode_steps=1000,
    reward_threshold=200.0,
)