"""
init file

Registers the DualCartPoleEnv environment.
"""

from gymnasium.envs.registration import register

"""
Version control DualCartPole:

    - v0: Added dynamics to include the second pole
    - v1: Fixed a bug that the "high" was 4 dim, should be 6
    - v2: Display the shorter pole in front of the long one
    - v3: Altered the initial conditions to [1 0 0 0 0 0]
    - v4: Changed it back to random conditions, found out the randomness it better for training
"""
register(
    id="DualCartPole-v4",
    entry_point="environments.cartpole_dualpendulum:DualCartPoleEnv",
)


"""
Version control DualCartPoleVec:

    - v0: Added dynamics to include the second pole
    - v2: Fixed a bug that the "high" was 4 dim, should be 6
          Display the shorter pole in front of the long one
    - v3: Altered the initial conditions to [1 0 0 0 0 0]
    - v4: Changed it back to random conditions, found out the randomness it better for training
"""
register(
    id="DualCartPoleVec-v4",
    entry_point="environments.cartpole_dualpendulum:DualCartPoleVectorEnv",
)
