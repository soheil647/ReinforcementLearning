import gym
import numpy as np


def extract_environment_specs(env):
    # Determine the type and dimension of the action space
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_type = 'discrete'
        dim_action = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        action_type = 'continuous'
        dim_action = env.action_space.shape[0]
    else:
        raise ValueError("Unsupported action space type")

    # Initialize defaults for image and vector dimensions
    dim_obs_img = None
    dim_obs_vec = 0

    # Check the structure of the observation space
    if isinstance(env.observation_space, gym.spaces.Tuple):
        for space in env.observation_space.spaces:
            if isinstance(space, gym.spaces.Box):
                if len(space.shape) == 3:  # This is likely an image (C, H, W)
                    dim_obs_img = space.shape
                else:
                    dim_obs_vec += np.prod(space.shape)  # Sum dimensions of all vector spaces
    elif isinstance(env.observation_space, gym.spaces.Box):
        if len(env.observation_space.shape) == 3:
            dim_obs_img = env.observation_space.shape  # It's an image
        else:
            dim_obs_vec = np.prod(env.observation_space.shape)  # It's a vector

    return dim_obs_img, dim_obs_vec, dim_action, action_type
