import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def smooth(scalars, weight=0.6):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Append smoothed value to the list
        last = smoothed_val  # Set the last to the smoothed value
    return smoothed


# Example of calculating standard deviation for the filled area
def calculate_bounds(values, weight=0.6):
    values_smoothed = smooth(values, weight)
    std_dev = np.std(values)  # Standard deviation of original values
    upper_bound = [v + std_dev for v in values_smoothed]
    lower_bound = [v - std_dev for v in values_smoothed]
    return upper_bound, lower_bound

def rolling_min_max(values, window):
    # Convert the numpy array to a pandas Series
    series = pd.Series(values)

    # Calculate rolling minimum and maximum
    rolling_min = series.rolling(window=window, min_periods=1).min()
    rolling_max = series.rolling(window=window, min_periods=1).max()

    # Return as numpy arrays
    return rolling_min.to_numpy(), rolling_max.to_numpy()


# Load your data, smooth it and plot
# ea1 = event_accumulator.EventAccumulator('RL/tensorboard/DQN/data_2024-06-03_21-30-20_batch_size-32_seed-2')
# ea1 = event_accumulator.EventAccumulator('RL/tensorboard/DDPG/data_2024-06-01_16-03-39_batch_size-64_seed-1')
ea1 = event_accumulator.EventAccumulator('RL/tensorboard/DQN/data_2024-06-03_21-04-21_batch_size-32_seed-2')
ea1.Reload()

scalar_list1 = ea1.Scalars('Performance/episodic_return')
steps1 = [s.step for s in scalar_list1]
values1 = [s.value for s in scalar_list1]
smoothed_values1 = smooth(values1, weight=0.60)
upper_bound1, lower_bound1 = calculate_bounds(values1, weight=0.60)

# ea2 = event_accumulator.EventAccumulator('RL/tensorboard/DDQN/data_2024-06-03_20-33-42_batch_size-32_seed-4')
# ea2 = event_accumulator.EventAccumulator('RL/tensorboard/SAC/data_2024-06-02_11-22-12_batch_size-128_seed-3')
ea2 = event_accumulator.EventAccumulator('RL/tensorboard/DDQN/data_GridWorld_2024-06-04_00-20-31')
ea2.Reload()

scalar_list2 = ea2.Scalars('Performance/episodic_return')
steps2 = [s.step for s in scalar_list2]
values2 = [s.value for s in scalar_list2]
smoothed_values2 = smooth(values2, weight=0.60)
upper_bound2, lower_bound2 = calculate_bounds(values2, weight=0.60)

mean_top_20_percent1 = np.mean(sorted([s.value for s in scalar_list1])[int(np.ceil(0.90 * len(scalar_list1))):])
mean_top_20_percent2 = np.mean(sorted([s.value for s in scalar_list2])[int(np.ceil(0.90 * len(scalar_list2))):])
print("Mean of the top 2% values 1:", mean_top_20_percent1)
print("Mean of the top 2% values 2:", mean_top_20_percent2)


window_size = 50

# Calculate rolling min and max for DQN
min_vals1, max_vals1 = rolling_min_max(smoothed_values1, window_size)
min_vals2, max_vals2 = rolling_min_max(smoothed_values2, window_size)


plt.figure(figsize=(10, 5))
plt.plot(steps1, smoothed_values1, label='DQN')
plt.fill_between(steps1, min_vals1, max_vals1, color='blue', alpha=0.3)
plt.plot(steps2, smoothed_values2, label='DDQN')
plt.fill_between(steps2, min_vals2, max_vals2, color='red', alpha=0.3)

plt.xlabel('Episodes')
plt.ylabel('Episodic Return')
plt.title('Smoothed return over episodes for DQN vs DDQN for GridWorld ')
plt.legend()
plt.grid(True)
plt.savefig('DDQN_DQN_GridWorld.png')
plt.show()