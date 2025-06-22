"""
Main File

This file is the orchestrator of the project. 
It trains the DQN and plots results / training progress.
"""

import gymnasium as gym
from DQN import AgentDQN
import environments.cartpole_dualpendulum
import torch as t
import numpy as np


class RLDualInvertedPendulum():
    """
    Main class for the entire project
    """
    def __init__(self, env):
        self.env = gym.make(env, render_mode="rgb_array")
        input_dim = self.env.observation_space.shape[0]
        output_dim = 2
        self.RL_Agent = AgentDQN(input_dim, output_dim, self.env)

    
    def train_DQN(self):
        # Train DQN
        self.RL_Agent.train()
        self.RL_Agent.save_model()


    def simulate_results(self, model_path):
        # Load trained model
        self.RL_Agent.policy_nn = t.load(model_path)
        self.RL_Agent.policy_nn.eval()

        # Rebuild the environment for visualization
        env = gym.make("DualCartPole-v4", render_mode="human")

        # Override the state
        env.reset()
        env.unwrapped.state = [0.3, 0., 0., 0., 0., 0.]
        obs = env.unwrapped.state

        # Result arrays
        timestep = 0.02
        time = np.array([])
        x = np.array([])
        theta1 = np.array([])
        theta2 = np.array([])

        step = 0
        while step < 20:  # simulate for 20 seconds
            state_tensor = t.FloatTensor(obs).unsqueeze(0)
            with t.no_grad():
                q_values = self.RL_Agent.policy_nn(state_tensor)
                action = t.argmax(q_values).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Log
            np.append(time, step)
            np.append(x, obs[0])
            np.append(theta1, obs[2])
            np.append(theta2, obs[4])

            step += timestep
            if done:
                break


if __name__ == '__main__':
    # Choose desired environment
    env = "DualCartPole-v4"

    # Select mode (train or simulate)
    mode = 'simulate'
    inv_pend = RLDualInvertedPendulum(env)
    
    if mode == 'train':
        inv_pend.train_DQN()
    
    elif mode == 'simulate':
        model = 'Models/DQN.pth'
        inv_pend.simulate_results(model)
