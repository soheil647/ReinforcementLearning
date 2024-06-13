import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import ChromaPalette as cp
import numpy as np

plt.rcParams['font.sans-serif']="Times New Roman"
plt.rcParams.update({'font.size': 35})

# Data
labels = ['Benchmark', 'DQN', 'DDQN']
values = [0.93, 0.83, 0.07]
steps = [0, 1000, 1000]

x = np.arange(len(labels))  # the label locations
colors = cp.chroma_palette.color_palette("Candies", 5)

# Bar width
width = 0.35  # the width of the bars

# Creating the bar chart
fig, ax1 = plt.subplots(figsize=(12, 8))

# Bar chart for values
bars1 = ax1.bar(x - width/2, values, width, label='Reward', color=colors[1])

# Make the y-axis label, ticks and tick labels match the first data series.
ax1.set_ylabel('Reward', color=colors[1])
ax1.tick_params(axis='y', labelcolor=colors[1])
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# Create a second y-axis to share the same x-axis
ax2 = ax1.twinx()

# Bar chart for steps on the second y-axis
bars2 = ax2.bar(x + width/2, steps, width, label='Steps', color=colors[3], alpha=0.6)

ax2.set_ylabel('Steps', color=colors[3])
ax2.tick_params(axis='y', labelcolor=colors[3])

# Title and legend
ax1.set_title('Reward and Steps by Method')
fig.tight_layout()  # Adjust the layout to make room for the second y-axis
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.savefig('GridWorldEnvResults.pdf')
plt.show()