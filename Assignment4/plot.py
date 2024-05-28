import numpy as np
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

# Load TensorBoard logs for the first log directory
ea1 = event_accumulator.EventAccumulator('tensorboard/sac/data_2024-05-21_10-38-16_alpha-0.2_batch_size-128_seed-3')
# ea1 = event_accumulator.EventAccumulator('tensorboard/ddpg/data_2024-05-21_09-54-13_lr-0.001,0.005_batch_size-32_seed-3')
ea1.Reload()

# Extract scalar data
scalar_list1 = ea1.Scalars('Performance/episodic_return')
steps1 = [s.step for s in scalar_list1]
values1 = [s.value for s in scalar_list1]
smoothed_values1 = smooth(values1, weight=0.95)

# Load TensorBoard logs for the second log directory
# ea2 = event_accumulator.EventAccumulator('tensorboard/sac/data_2024-05-20_12-08-39_alpha-0.2_batch_size-128_seed-1')
ea2 = event_accumulator.EventAccumulator('tensorboard/sac/data_2024-05-20_16-09-13_alpha-0.2_batch_size-128_seed-2')
ea2.Reload()

# Extract scalar data
scalar_list2 = ea2.Scalars('Performance/episodic_return')
steps2 = [s.step for s in scalar_list2]
values2 = [s.value for s in scalar_list2]
smoothed_values2 = smooth(values2, weight=0.95)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(steps1, smoothed_values1, label='Smoothed Episodic Return Run 1')
plt.plot(steps2, smoothed_values2, label='Smoothed Episodic Return Run 2', linestyle='--')  # Different linestyle for distinction
plt.xlabel('Steps')
plt.ylabel('Episodic Return')
plt.title('Smoothed Episodic Return Over Time for Multiple Runs')
plt.legend()
plt.grid(True)
plt.savefig('SAC_smoothed_episodic_return_comparison.png')  # Save the plot
plt.show()
