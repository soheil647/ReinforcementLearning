import torch


def dqn_config(observation_space, action_space, obs_type):
    config = {
        'dim_obs': observation_space,  # Q network input
        # 'dim_obs_vec': observation_space_vector,  # Q network input
        'obs_type': obs_type,
        'dim_action': action_space,  # Q network output
        # 'action_type': action_type,  # Q network output
        'dims_hidden_neurons': (64, 64),  # Q network hidden
        'lr': 0.001,  # learning rate
        'C': 20,  # copy steps
        'discount': 0.99,  # discount factor
        'batch_size': 32,
        'replay_buffer_size': 100000,
        'eps_min': 0.01,
        'eps_max': 1.0,
        'eps_len': 200,
        'seed': 2,
        'max_episode': 200,
        'device': torch.device('cuda:0')
    }
    return config


def ddqn_config(observation_space, action_space, obs_type):
    config = {
        'dim_obs': observation_space,  # Q network input
        # 'dim_obs_vec': observation_space_vector,  # Q network input
        'obs_type': obs_type,
        'dim_action': action_space,  # Q network output
        # 'action_type': action_type,  # Q network output
        'dims_hidden_neurons': (128, 128),  # Q network hidden
        'lr': 0.001,  # learning rate
        'C': 30,  # copy steps
        'discount': 0.99,  # discount factor
        'batch_size': 64,
        'replay_buffer_size': 200000,
        'eps_min': 0.01,
        'eps_max': 1.0,
        'eps_len': 1000,
        'seed': 4,
        'max_episode': 1000,
        'device': torch.device('cuda:0')
    }
    return config


def ddpg_config(observation_space, action_space, obs_type):
    config = {
        'dim_obs': observation_space,
        'dim_action': action_space,
        'obs_type': obs_type,
        'dims_hidden_neurons': (600, 200),
        'lr_actor': 0.001,
        'lr_critic': 0.005,
        # 'smooth': 0.99,
        'smooth': 0.001,
        'discount': 0.99,
        'sig': 0.01,
        'batch_size': 64,
        'replay_buffer_size': 200000,
        'seed': 1,
        'max_episode': 2000,
        'device': torch.device('cuda:0')
    }
    return config


def sac_config(observation_space, action_space, obs_type):
    config = {
        'dim_obs': observation_space,
        'dim_action': action_space,
        'dims_hidden_neurons': (120, 120),
        'obs_type': obs_type,
        'lr': 0.001,
        # 'smooth': 0.99,
        'smooth': 0.001,
        'discount': 0.99,
        'alpha': 0.2,
        'batch_size': 128,
        'replay_buffer_size': 20000,
        'seed': 3,
        'max_episode': 1000,
        'device': torch.device('cuda:0')
    }
    return config
