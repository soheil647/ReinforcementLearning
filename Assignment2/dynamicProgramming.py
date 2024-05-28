import numpy as np

# Define the state-action reward function r(s, a)
reward = np.array([
    [0.0, 0.2],
    [0.0, 0.2],
    [0.0, 0.2],
    [0.0, 0.2],
    [1.0, 0.2]
])

# Define the state transition probabilities p(s'|s, a)
transition_a0 = np.array([
    [0.8, 0.2, 0.0, 0.0, 0.0],
    [0.0, 0.8, 0.2, 0.0, 0.0],
    [0.0, 0.0, 0.8, 0.2, 0.0],
    [0.0, 0.0, 0.0, 0.8, 0.2],
    [0.0, 0.0, 0.0, 0.0, 1.0]
])

transition_a1 = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0],
    [0.0, 0.9, 0.1, 0.0, 0.0],
    [0.0, 0.0, 0.9, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.9, 0.1],
    [0.0, 0.0, 0.0, 0.0, 1.0]
])

# Number of states and actions
num_states = 5
num_actions = 2

# Discount factor
gamma = 0.95

# Value iteration algorithm
iterations = 1000  # Maximum number of iterations
threshold = 1e-6  # Convergence threshold

# Initialize Q-values to zero
Q = np.zeros((num_states, num_actions))

for i in range(iterations):
    Q_prev = Q.copy()
    for s in range(num_states):
        for a in range(num_actions):
            if a == 0:
                Q[s, a] = np.sum(transition_a0[s] * (reward[s, a] + gamma * np.max(Q_prev, axis=1)))
            else:
                Q[s, a] = np.sum(transition_a1[s] * (reward[s, a] + gamma * np.max(Q_prev, axis=1)))
    if np.max(abs(Q - Q_prev)) < threshold:
        break

    # if i >= 3:
    #     exit()

print(Q)