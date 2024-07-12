import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


# Define the neural network model using PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the RL agent
class RLAgent:
    def __init__(self, env, hidden_size=24, gamma=0.95, epsilon_decay=0.997, epsilon_min=0.01):
        self.env = env
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.hidden_size = hidden_size
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=1000000)
        self.gamma = gamma

        # Create the neural network model
        self.model = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        self.target_model = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def select_action(self, observation):
        if np.random.rand() > self.epsilon:
            with torch.no_grad():
                observation = torch.FloatTensor(observation).unsqueeze(0)
                return torch.argmax(self.model(observation)).item()
        else:
            return np.random.randint(self.output_size)

    def remember(self, prev_obs, action, reward, observation, done):
        self.memory.append((prev_obs, action, reward, observation, done))

    def experience_replay(self, batch_size=20):
        if len(self.memory) < batch_size:
            return

        batch = np.random.choice(len(self.memory), batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())