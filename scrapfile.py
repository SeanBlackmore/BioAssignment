import gymnasium as gym
import environments.cartpole_dualpendulum

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch as t

from plot_results import plot_time_responses


def train():
    env = gym.make("DualCartPole-v4", render_mode="rgb_array")
    print(env.observation_space.shape)  # Should print (6,)

    # Play around with a custom reward function
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


    class RewardCallback(BaseCallback):
        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    print(f"Episode reward: {info['episode']['r']}")
            return True


    # Set the wrapper
    env = CustomReward(env)
    env = Monitor(env)

    # Create and train the model
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2000000, callback=RewardCallback())
    model.save("dqn_cartpole")

    # Delete and reload the model
    del model


def sim():
    env = gym.make("DualCartPole-v4", render_mode="human")  # "human" for real-time visualization
    model = DQN.load("Models/dqn_cartpole")

    # episodes = 0
    # while episodes < 10:
    obs, _ = env.reset()

    # Initialize result arrays
    timestep = 0.02
    time = np.array([])
    x = np.array([])
    theta1 = np.array([])
    theta2 = np.array([])

    # Override internal state
    env.unwrapped.state = [1., 0., 0., 0., 0., 0.]

    # Also update observation if needed
    obs = env.unwrapped.state

    done = False

    step = 0
    while step < 20:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Append the results
        time = np.append(time, step)
        x = np.append(x, obs[0])
        theta1 = np.append(theta1, obs[2])
        theta2 = np.append(theta2, obs[4])

        step += timestep

        # episodes += 1

    # Plot results
    plot_time_responses(time, x, theta1, theta2)

    env.close()

if __name__ == "__main__":
    # train()

    sim()