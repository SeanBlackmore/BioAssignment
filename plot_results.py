"""
Plot Results File

File with functions for plotting the following:
    - Time responses RL controller
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_process():
    pass


def plot_time_responses(t, x, theta1, theta2):
    """
    Plots the time response of the RL agent controlling the dual inverted pendulum.
    Generated plots:
        - Cart position
        - theta_1
        - theta_2
    """

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True, layout="constrained")  # width=10, height=6

    # Plot 1: Cart Position
    axs[0].plot(t, x, label="Cart Position (x)")
    axs[0].axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Target")
    axs[0].set_ylabel("x [m]")
    axs[0].set_title("Cart Position")
    axs[0].grid(True)
    axs[0].legend()

    # Plot 2: Pole 1 Angle (short pole)
    axs[1].plot(t, theta1 * 180 / np.pi, label="Theta 1 (short pole)", color='tab:orange')
    axs[1].axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Target")
    axs[1].set_ylabel("θ₁ [deg]")
    axs[1].set_title("Short Pole Angle")
    axs[1].grid(True)
    axs[1].legend()

    # Plot 3: Pole 2 Angle (long pole)
    axs[2].plot(t, theta2 * 180 / np.pi, label="Theta 2 (long pole)", color='tab:green')
    axs[2].axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Target")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("θ₂ [deg]")
    axs[2].set_title("Long Pole Angle")
    axs[2].grid(True)
    axs[2].legend()

    plt.suptitle("Dual Inverted Pendulum Dynamics", fontsize=14)
    plt.show()