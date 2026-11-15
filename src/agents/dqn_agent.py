
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# --------------------------------------------------
# 1. DQN Neural Network
# --------------------------------------------------

class QNetwork(nn.Module):

    def __init__(self, state_dim=21, action_dim=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.network(state)


# --------------------------------------------------
# 2. Experience Replay Buffer
# --------------------------------------------------

class ReplayBuffer:

    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.memory)


# --------------------------------------------------
# 3. DQN Agent
# --------------------------------------------------

class DQNAgent:

    def __init__(
        self,
        state_dim=21,
        action_dim=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        memory_capacity=10000,
        target_update=100
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device("cpu")

        # Main network
        self.policy_net = QNetwork(
            state_dim, action_dim
        ).to(self.device)

        # Target network
        self.target_net = QNetwork(
            state_dim, action_dim
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        # Replay buffer
        self.memory = ReplayBuffer(memory_capacity)

        self.learning_steps = 0

    # ----------------------------------------------
    # Epsilon-greedy action selection
    # ----------------------------------------------

    def select_action(self, state, training=True):

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return q_values.argmax(dim=1).item()

    # ----------------------------------------------
    # Store experience
    # ----------------------------------------------

    def remember(self, state, action, reward, next_state, done):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    # ----------------------------------------------
    # Train the policy network
    # ----------------------------------------------

    def train_step(self):

        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = (
            self.memory.sample(self.batch_size)
        )

        states = torch.tensor(
            states, dtype=torch.float32, device=self.device
        )

        actions = torch.tensor(
            actions, dtype=torch.long, device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        )

        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )

        dones = torch.tensor(
            dones, dtype=torch.float32, device=self.device
        )

        # Q(s, a) predicted by the policy network
        current_q_values = self.policy_net(states).gather(
            1, actions
        ).squeeze(1)

        # Target Q-values
        with torch.no_grad():

            next_q_values = self.target_net(
                next_states
            ).max(dim=1)[0]

            target_q_values = (
                rewards
                + self.gamma * next_q_values * (1 - dones)
            )

        # Loss
        loss = nn.SmoothL1Loss()(
            current_q_values,
            target_q_values
        )

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        # Update target network periodically
        self.learning_steps += 1

        if self.learning_steps % self.target_update == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        return loss.item()

    # ----------------------------------------------
    # Epsilon decay
    # ----------------------------------------------

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


# --------------------------------------------------
# 4. Basic Component Test
# --------------------------------------------------

if __name__ == "__main__":

    print("=== DQN COMPONENT TEST ===")

    agent = DQNAgent()

    test_state = np.random.random(21).astype(np.float32)
    action = agent.select_action(test_state)

    print("State dimension:", agent.state_dim)
    print("Action dimension:", agent.action_dim)
    print("Selected action:", action)
    print("Initial epsilon:", agent.epsilon)

    # Add synthetic experiences to test replay and learning
    for _ in range(100):

        state = np.random.random(21).astype(np.float32)
        next_state = np.random.random(21).astype(np.float32)

        action = random.randrange(4)    
        reward = random.uniform(-2, 0)
        done = False

        agent.remember(
            state, action, reward, next_state, done
        )

    print("Replay buffer size:", len(agent.memory))

    loss = agent.train_step()

    print("Training loss:", loss)
    print("Learning steps:", agent.learning_steps)

    print("\nDQN component test completed.")
