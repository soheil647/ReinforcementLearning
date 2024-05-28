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

# Learning rate
alpha = 0.1

# Epsilon for the epsilon-greedy policy
epsilon = 0.1

iterations = 1000000  # Maximum number of iterations

Q = np.zeros((num_states, num_actions))

# Simulate the process of Q-learning
for i in range(iterations):
    # Start from a random state
    s = np.random.choice(num_states)

    # Choose action using epsilon-greedy policy
    if np.random.rand() < epsilon:
        a = np.random.choice(num_actions)
    else:
        a = np.argmax(Q[s])

    # Simulate taking action 'a' in state 's' and landing in a new state 's_prime' with probability of state transition
    if a == 0:
        s_prime = np.random.choice(num_states, p=transition_a0[s])
    else:
        s_prime = np.random.choice(num_states, p=transition_a1[s])

    # Receive the reward for the transition
    r = reward[s, a]

    # Q-learning update
    Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[s_prime]) - Q[s, a])

print(Q)
