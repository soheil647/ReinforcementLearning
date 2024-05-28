import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import datetime


env = gym.make('CartPole-v1',render_mode="human")
dqn_model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    train_freq=256,
    gradient_steps=128,
    gamma=0.99,
    exploration_fraction=0.16,
    exploration_final_eps=0.04,
    target_update_interval=10,
    learning_starts=1000,
    buffer_size=100000,
    batch_size=64,
    learning_rate=2.3e-3,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log='tensorboard/baseline_dqn_{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now()),
    seed=0,
)
dqn_model.learn(total_timesteps=500, log_interval=5)

# evaluate
mean_reward, std_reward = evaluate_policy(dqn_model, dqn_model.get_env(), deterministic=True, n_eval_episodes=20)
print(f"evaluation mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# dqn_model save & load
# dqn_model.save("deepq_cartpole")
# del dqn_model  # remove to demonstrate saving and loading
# model = DQN.load("deepq_cartpole")

# test model
# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()


