"""
Deep Q Learning File

This file holds my implementation of a Deep Q learning algorithm.
Inspiration is taken from https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae.
"""

import gymnasium as gym
import numpy as np
import torch as t
import random as rd
from collections import deque


class CustomReward(gym.RewardWrapper):
        """
        Custom reward function for the cart pole environment
        """
        def __init__(self, env):
            super().__init__(env)
        
        def reward(self, reward):
            # Get the current state
            state = self.unwrapped.state

            # Unpack the state
            x, x_dot, theta1, theta1_dot, theta2, theta2_dot = state

            # Reward
            reward = (
                1.0 - (abs(x)) - 0.01 * (x_dot) - 0.001
            )
            return np.clip(reward, -1., 1.)


class AgentDQN():
    """
    The DQN class
    """
    def __init__(self, input_dim, output_dim, env):
        # Declare all the hyperparameters here
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update_freq = 1000
        self.memory_size = 10000
        self.episodes = 1000

        # Set the environment and overwrite the reward function
        self.env = CustomReward(env)

        # Set up model architecture
        nn_architecture = t.nn.Sequential(
            t.nn.Linear(input_dim, 64),
            t.nn.ReLU(),
            t.nn.Linear(64, 128),
            t.nn.ReLU(),
            t.nn.Linear(128, 128),
            t.nn.ReLU(),
            t.nn.Linear(128, output_dim)
        )

        # Initialize networks
        self.policy_nn = nn_architecture
        self.target_nn = nn_architecture

        self.optim = t.optim.Adam(self.policy_nn.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=self.memory_size)



    def select_action(self, state):
        """
        Selects an action using the epsilon greedy approach
        """
        if rd.random() < self.epsilon:
            return self.env.action_space.sample()   # Explore
        else:
            state = t.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_nn(state)
            return t.argmax(q_values).item()        # Exploit

    
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        batch = rd.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = t.FloatTensor(state_batch)
        action_batch = t.LongTensor(action_batch).unsqueeze(1)
        reward_batch = t.FloatTensor(reward_batch)
        next_state_batch = t.FloatTensor(next_state_batch)
        done_batch = t.FloatTensor(done_batch)

        # Compute Q-values for current states
        q_values = self.policy_nn(state_batch).gather(1, action_batch).squeeze()

        # Compute target Q-values using the target network
        with t.no_grad():
            max_next_q_values = self.target_nn(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)

        loss = t.nn.MSELoss()(q_values, target_q_values)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


    def train(self):
        # Main training loop
        rewards_per_episode = []
        steps_done = 0

        for episode in range(self.episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select action
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store transition in memory
                self.memory.append((state, action, reward, next_state, done))

                # Update state
                state = next_state
                episode_reward += reward

                # Optimize model
                self.optimize()

                # Update target network periodically
                if steps_done % self.target_update_freq == 0:
                    self.target_nn.load_state_dict(self.policy_nn.state_dict())

                steps_done += 1

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

            rewards_per_episode.append(episode_reward)
            print(f"Episode {episode + 1}/{self.episodes}, Reward: {episode_reward}, Epsilon: {self.epsilon:.3f}")
    

    def save_model(self):
        t.save(self.policy_nn, 'Models/DQN.pth')
        print(f"Saved model to 'Models/DQN.pth'!")
