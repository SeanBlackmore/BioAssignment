"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space


class DualCartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards
    Since the goal is to keep the pole upright for as long as possible, by default, a reward of `+1` is given for every step taken, including the termination step. The default reward threshold is 500 for v1 and 200 for v0 due to the time limit on the environment.

    If `sutton_barto_reward=True`, then a reward of `0` is awarded for every non-terminating step and `-1` for the terminating step. As a result, the reward threshold is 0 for v0 and v1.

    ## Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End
    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    Cartpole only has `render_mode` as a keyword for `gymnasium.make`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CartPole-v1", render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
    >>> env.reset(seed=123, options={"low": -0.1, "high": 0.1})  # default low=-0.05, high=0.05
    (array([ 0.03647037, -0.0892358 , -0.05592803, -0.06312564], dtype=float32), {})

    ```

    | Parameter               | Type       | Default                 | Description                                                                                   |
    |-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
    | `sutton_barto_reward`   | **bool**   | `False`                 | If `True` the reward function matches the original sutton barto implementation                |

    ## Vectorized environment

    To increase steps per seconds, users can use a custom vector environment or with an environment vectorizor.

    ```python
    >>> import gymnasium as gym
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="vector_entry_point")
    >>> envs
    CartPoleVectorEnv(CartPole-v1, num_envs=3)
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
    >>> envs
    SyncVectorEnv(CartPole-v1, num_envs=3)

    ```

    ## Version History
    * v1: `max_time_steps` raised to 500.
        - In Gymnasium `1.0.0a2` the `sutton_barto_reward` argument was added (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/790))
    * v0: Initial versions release.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self, sutton_barto_reward: bool = False, render_mode: Optional[str] = None
    ):
        self._sutton_barto_reward = sutton_barto_reward

        self.gravity = 9.81 # m/s^2
        self.masscart = 5   # kg
        # self.masspole = 0.1
        self.masspole1 = 0.1    # kg, Added
        self.masspole2 = 0.2    # kg, Added
        self.total_mass = self.masspole1 + self.masspole2 + self.masscart
        self.length1 = 0.5  # actually half the pole's length
        self.length2 = 1    # Added
        #self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.Z = 4 * self.masscart + self.masspole1 + self.masspole2

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds. CHECK THIS
        high = np.array([
            self.x_threshold * 2,               # x
            np.inf,                             # x_dot
            self.theta_threshold_radians * 2,   # theta1
            np.inf,                             # theta_dot1
            self.theta_threshold_radians * 2,   # theta2
            np.inf                              # theta_dot2
        ], dtype=np.float32)


        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None

        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        # x, x_dot, theta, theta_dot = self.state       
        x, x_dot, theta1, theta_dot1, theta2, theta_dot2 = self.state   # Added
        force = self.force_mag if action == 1 else -self.force_mag
        # costheta = np.cos(theta)
        # sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        # temp = (
        #     force + self.polemass_length * np.square(theta_dot) * sintheta
        # ) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta * temp) / (
        #     self.length
        #     * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        # )
        # xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x_acc = (4 / self.Z) * force \
                - ((3 * self.masspole1 * self.gravity) / self.Z) * theta1 \
                - ((3 * self.masspole2 * self.gravity) / self.Z) * theta2
        theta1acc = (-3 / (self.Z * self.length1)) * force \
                + ((3 * self.gravity * (self.Z + 3 * self.masspole1)) / (4 * self.Z * self.length1)) * theta1 \
                + ((9 * self.masspole2 * self.gravity) / (4 * self.Z * self.length1)) * theta2
        theta2acc = (-3 / (self.Z * self.length2)) * force \
                + ((9 * self.masspole1 * self.gravity) / (4 * self.Z * self.length2)) * theta1 \
                + ((3 * self.gravity * (self.Z + 3 * self.masspole2)) / (4 * self.Z * self.length2)) * theta2

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * x_acc
            theta1 = theta1 + self.tau * theta_dot1         # Modified
            theta_dot1 = theta_dot1 + self.tau * theta1acc  # Modified
            theta2 = theta2 + self.tau * theta_dot2         # Modified
            theta_dot2 = theta_dot2 + self.tau * theta2acc  # Modified
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * x_acc
            x = x + self.tau * x_dot
            theta_dot1 = theta_dot1 + self.tau * theta1acc  # Modified
            theta1 = theta1 + self.tau * theta_dot1         # Modified
            theta_dot2 = theta_dot2 + self.tau * theta2acc  # Modified
            theta2 = theta2 + self.tau * theta_dot2         # Modified

        self.state = np.array((x, x_dot, theta1, theta_dot1, theta2, theta_dot2), dtype=np.float64)     # Modified

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta1 < -self.theta_threshold_radians       # Modified
            or theta1 > self.theta_threshold_radians        # Modified
            or theta2 < -self.theta_threshold_radians       # Modified
            or theta2 > self.theta_threshold_radians        # Modified
        )

        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0

            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

            reward = -1.0 if self._sutton_barto_reward else 0.0

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(6,))
        # self.state = np.array([1., 0., 0., 0., 0., 0.], dtype=np.float32)       # Modified
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    # def render(self):
    #     if self.render_mode is None:
    #         assert self.spec is not None
    #         gym.logger.warn(
    #             "You are calling render method without specifying any render mode. "
    #             "You can specify the render_mode at initialization, "
    #             f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
    #         )
    #         return

    #     try:
    #         import pygame
    #         from pygame import gfxdraw
    #     except ImportError as e:
    #         raise DependencyNotInstalled(
    #             'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
    #         ) from e

    #     if self.screen is None:
    #         pygame.init()
    #         if self.render_mode == "human":
    #             pygame.display.init()
    #             self.screen = pygame.display.set_mode(
    #                 (self.screen_width, self.screen_height)
    #             )
    #         else:  # mode == "rgb_array"
    #             self.screen = pygame.Surface((self.screen_width, self.screen_height))
    #     if self.clock is None:
    #         self.clock = pygame.time.Clock()

    #     world_width = self.x_threshold * 2
    #     scale = self.screen_width / world_width
    #     polewidth = 10.0
    #     polelen = scale * (2 * self.length)
    #     cartwidth = 50.0
    #     cartheight = 30.0

    #     if self.state is None:
    #         return None

    #     x = self.state

    #     self.surf = pygame.Surface((self.screen_width, self.screen_height))
    #     self.surf.fill((255, 255, 255))

    #     l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
    #     axleoffset = cartheight / 4.0
    #     cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
    #     carty = 100  # TOP OF CART
    #     cart_coords = [(l, b), (l, t), (r, t), (r, b)]
    #     cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
    #     gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
    #     gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

    #     l, r, t, b = (
    #         -polewidth / 2,
    #         polewidth / 2,
    #         polelen - polewidth / 2,
    #         -polewidth / 2,
    #     )

    #     pole_coords = []
    #     for coord in [(l, b), (l, t), (r, t), (r, b)]:
    #         coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
    #         coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
    #         pole_coords.append(coord)
    #     gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
    #     gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

    #     gfxdraw.aacircle(
    #         self.surf,
    #         int(cartx),
    #         int(carty + axleoffset),
    #         int(polewidth / 2),
    #         (129, 132, 203),
    #     )
    #     gfxdraw.filled_circle(
    #         self.surf,
    #         int(cartx),
    #         int(carty + axleoffset),
    #         int(polewidth / 2),
    #         (129, 132, 203),
    #     )

    #     gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

    #     self.surf = pygame.transform.flip(self.surf, False, True)
    #     self.screen.blit(self.surf, (0, 0))
    #     if self.render_mode == "human":
    #         pygame.event.pump()
    #         self.clock.tick(self.metadata["render_fps"])
    #         pygame.display.flip()

    #     elif self.render_mode == "rgb_array":
    #         return np.transpose(
    #             np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
    #         )

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x, _, theta1, _, theta2, _ = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # Cart
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x * scale + self.screen_width / 2.0
        carty = 100
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        # Function to draw a single pole
        def draw_pole(theta, length, color):
            polelen = scale * (2 * length)  # Full length
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole_coords = []
            for coord in [(l, b), (l, t), (r, t), (r, b)]:
                rotated = pygame.math.Vector2(coord).rotate_rad(-theta)
                rotated = (rotated[0] + cartx, rotated[1] + carty + axleoffset)
                pole_coords.append(rotated)
            gfxdraw.aapolygon(self.surf, pole_coords, color)
            gfxdraw.filled_polygon(self.surf, pole_coords, color)

        # Draw both poles with different colors
        draw_pole(theta2, self.length2, (120, 180, 255))  # Light blue
        draw_pole(theta1, self.length1, (202, 152, 101))  # Brown
        

        # Axle circle
        gfxdraw.aacircle(self.surf, int(cartx), int(carty + axleoffset), int(polewidth / 2), (129, 132, 203))
        gfxdraw.filled_circle(self.surf, int(cartx), int(carty + axleoffset), int(polewidth / 2), (129, 132, 203))

        # Ground line
        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        # Flip vertically
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))


    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


# class DualCartPoleVectorEnv(VectorEnv):
#     metadata = {
#         "render_modes": ["rgb_array"],
#         "render_fps": 50,
#         "autoreset_mode": AutoresetMode.NEXT_STEP,
#     }

#     def __init__(
#         self,
#         num_envs: int = 1,
#         max_episode_steps: int = 500,
#         render_mode: Optional[str] = None,
#         sutton_barto_reward: bool = False,
#     ):
#         self._sutton_barto_reward = sutton_barto_reward

#         self.num_envs = num_envs
#         self.max_episode_steps = max_episode_steps
#         self.render_mode = render_mode

#         self.gravity = 9.8
#         self.masscart = 1.0
#         self.masspole = 0.1
#         self.total_mass = self.masspole + self.masscart
#         self.length = 0.5  # actually half the pole's length
#         self.polemass_length = self.masspole * self.length
#         self.force_mag = 10.0
#         self.tau = 0.02  # seconds between state updates
#         self.kinematics_integrator = "euler"

#         self.state = None

#         self.steps = np.zeros(num_envs, dtype=np.int32)
#         self.prev_done = np.zeros(num_envs, dtype=np.bool_)

#         # Angle at which to fail the episode
#         self.theta_threshold_radians = 12 * 2 * math.pi / 360
#         self.x_threshold = 2.4

#         # Angle limit set to 2 * theta_threshold_radians so failing observation
#         # is still within bounds.
#         high = np.array(
#             [
#                 self.x_threshold * 2,
#                 np.inf,
#                 self.theta_threshold_radians * 2,
#                 np.inf,
#             ],
#             dtype=np.float32,
#         )

#         self.low = -0.05
#         self.high = 0.05

#         self.single_action_space = spaces.Discrete(2)
#         self.action_space = batch_space(self.single_action_space, num_envs)
#         self.single_observation_space = spaces.Box(-high, high, dtype=np.float32)
#         self.observation_space = batch_space(self.single_observation_space, num_envs)

#         self.screen_width = 600
#         self.screen_height = 400
#         self.screens = None
#         self.surf = None

#         self.steps_beyond_terminated = None

#     def step(
#         self, action: np.ndarray
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
#         assert self.action_space.contains(
#             action
#         ), f"{action!r} ({type(action)}) invalid"
#         assert self.state is not None, "Call reset before using step method."

#         x, x_dot, theta, theta_dot = self.state
#         force = np.sign(action - 0.5) * self.force_mag
#         costheta = np.cos(theta)
#         sintheta = np.sin(theta)

#         # For the interested reader:
#         # https://coneural.org/florian/papers/05_cart_pole.pdf
#         temp = (
#             force + self.polemass_length * np.square(theta_dot) * sintheta
#         ) / self.total_mass
#         thetaacc = (self.gravity * sintheta - costheta * temp) / (
#             self.length
#             * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
#         )
#         xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

#         if self.kinematics_integrator == "euler":
#             x = x + self.tau * x_dot
#             x_dot = x_dot + self.tau * xacc
#             theta = theta + self.tau * theta_dot
#             theta_dot = theta_dot + self.tau * thetaacc
#         else:  # semi-implicit euler
#             x_dot = x_dot + self.tau * xacc
#             x = x + self.tau * x_dot
#             theta_dot = theta_dot + self.tau * thetaacc
#             theta = theta + self.tau * theta_dot

#         self.state = np.stack((x, x_dot, theta, theta_dot))

#         terminated: np.ndarray = (
#             (x < -self.x_threshold)
#             | (x > self.x_threshold)
#             | (theta < -self.theta_threshold_radians)
#             | (theta > self.theta_threshold_radians)
#         )

#         self.steps += 1

#         truncated = self.steps >= self.max_episode_steps

#         if self._sutton_barto_reward:
#             reward = -np.array(terminated, dtype=np.float32)
#         else:
#             reward = np.ones_like(terminated, dtype=np.float32)

#         # Reset all environments which terminated or were truncated in the last step
#         self.state[:, self.prev_done] = self.np_random.uniform(
#             low=self.low, high=self.high, size=(4, self.prev_done.sum())
#         )
#         self.steps[self.prev_done] = 0
#         reward[self.prev_done] = 0.0
#         terminated[self.prev_done] = False
#         truncated[self.prev_done] = False

#         self.prev_done = np.logical_or(terminated, truncated)

#         return self.state.T.astype(np.float32), reward, terminated, truncated, {}

#     def reset(
#         self,
#         *,
#         seed: Optional[int] = None,
#         options: Optional[dict] = None,
#     ):
#         super().reset(seed=seed)
#         # Note that if you use custom reset bounds, it may lead to out-of-bound
#         # state/observations.
#         # -0.05 and 0.05 is the default low and high bounds
#         self.low, self.high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
#         self.state = self.np_random.uniform(
#             low=self.low, high=self.high, size=(4, self.num_envs)
#         )
#         self.steps_beyond_terminated = None
#         self.steps = np.zeros(self.num_envs, dtype=np.int32)
#         self.prev_done = np.zeros(self.num_envs, dtype=np.bool_)

#         return self.state.T.astype(np.float32), {}

#     def render(self):
#         if self.render_mode is None:
#             assert self.spec is not None
#             gym.logger.warn(
#                 "You are calling render method without specifying any render mode. "
#                 "You can specify the render_mode at initialization, "
#                 f'e.g. gym.make_vec("{self.spec.id}", render_mode="rgb_array")'
#             )
#             return

#         try:
#             import pygame
#             from pygame import gfxdraw
#         except ImportError:
#             raise DependencyNotInstalled(
#                 'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
#             )

#         if self.screens is None:
#             pygame.init()

#             self.screens = [
#                 pygame.Surface((self.screen_width, self.screen_height))
#                 for _ in range(self.num_envs)
#             ]

#         world_width = self.x_threshold * 2
#         scale = self.screen_width / world_width
#         polewidth = 10.0
#         polelen = scale * (2 * self.length)
#         cartwidth = 50.0
#         cartheight = 30.0

#         if self.state is None:
#             raise ValueError(
#                 "Cartpole's state is None, it probably hasn't be reset yet."
#             )

#         for x, screen in zip(self.state.T, self.screens):
#             assert isinstance(x, np.ndarray) and x.shape == (4,)

#             self.surf = pygame.Surface((self.screen_width, self.screen_height))
#             self.surf.fill((255, 255, 255))

#             l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
#             axleoffset = cartheight / 4.0
#             cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
#             carty = 100  # TOP OF CART
#             cart_coords = [(l, b), (l, t), (r, t), (r, b)]
#             cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
#             gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
#             gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

#             l, r, t, b = (
#                 -polewidth / 2,
#                 polewidth / 2,
#                 polelen - polewidth / 2,
#                 -polewidth / 2,
#             )

#             pole_coords = []
#             for coord in [(l, b), (l, t), (r, t), (r, b)]:
#                 coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
#                 coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
#                 pole_coords.append(coord)
#             gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
#             gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

#             gfxdraw.aacircle(
#                 self.surf,
#                 int(cartx),
#                 int(carty + axleoffset),
#                 int(polewidth / 2),
#                 (129, 132, 203),
#             )
#             gfxdraw.filled_circle(
#                 self.surf,
#                 int(cartx),
#                 int(carty + axleoffset),
#                 int(polewidth / 2),
#                 (129, 132, 203),
#             )

#             gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

#             self.surf = pygame.transform.flip(self.surf, False, True)
#             screen.blit(self.surf, (0, 0))

#         return [
#             np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
#             for screen in self.screens
#         ]

#     def close(self):
#         if self.screens is not None:
#             import pygame

#             pygame.quit()

class DualCartPoleVectorEnv(VectorEnv):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 50,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None,
        sutton_barto_reward: bool = False,
    ):
        self._sutton_barto_reward = sutton_barto_reward

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.gravity = 9.81
        self.masscart = 5.0
        self.masspole1 = 0.1
        self.masspole2 = 0.2
        self.length1 = 0.5
        self.length2 = 1.0
        self.Z = 4 * self.masscart + self.masspole1 + self.masspole2
        self.force_mag = 10.0
        self.tau = 0.02
        self.kinematics_integrator = "euler"

        self.state = None

        self.steps = np.zeros(num_envs, dtype=np.int32)
        self.prev_done = np.zeros(num_envs, dtype=bool)

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array([
            self.x_threshold * 2,
            np.inf,
            self.theta_threshold_radians * 2,
            np.inf,
            self.theta_threshold_radians * 2,
            np.inf,
        ], dtype=np.float32)

        self.low = -0.05
        self.high = 0.05

        self.single_action_space = spaces.Discrete(2)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta1, theta_dot1, theta2, theta_dot2 = self.state
        force = np.where(action == 1, self.force_mag, -self.force_mag)

        x_acc = (4 / self.Z) * force \
              - ((3 * self.masspole1 * self.gravity) / self.Z) * theta1 \
              - ((3 * self.masspole2 * self.gravity) / self.Z) * theta2

        theta1acc = (-3 / (self.Z * self.length1)) * force \
              + ((3 * self.gravity * (self.Z + 3 * self.masspole1)) / (4 * self.Z * self.length1)) * theta1 \
              + ((9 * self.masspole2 * self.gravity) / (4 * self.Z * self.length1)) * theta2

        theta2acc = (-3 / (self.Z * self.length2)) * force \
              + ((9 * self.masspole1 * self.gravity) / (4 * self.Z * self.length2)) * theta1 \
              + ((3 * self.gravity * (self.Z + 3 * self.masspole2)) / (4 * self.Z * self.length2)) * theta2

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * x_acc
            theta1 = theta1 + self.tau * theta_dot1
            theta_dot1 = theta_dot1 + self.tau * theta1acc
            theta2 = theta2 + self.tau * theta_dot2
            theta_dot2 = theta_dot2 + self.tau * theta2acc
        else:
            x_dot = x_dot + self.tau * x_acc
            x = x + self.tau * x_dot
            theta_dot1 = theta_dot1 + self.tau * theta1acc
            theta1 = theta1 + self.tau * theta_dot1
            theta_dot2 = theta_dot2 + self.tau * theta2acc
            theta2 = theta2 + self.tau * theta_dot2

        self.state = np.stack((x, x_dot, theta1, theta_dot1, theta2, theta_dot2))

        terminated = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta1 < -self.theta_threshold_radians)
            | (theta1 > self.theta_threshold_radians)
            | (theta2 < -self.theta_threshold_radians)
            | (theta2 > self.theta_threshold_radians)
        )

        self.steps += 1
        truncated = self.steps >= self.max_episode_steps

        if self._sutton_barto_reward:
            reward = -terminated.astype(np.float32)
        else:
            reward = np.ones_like(terminated, dtype=np.float32)

        # Reset environments that are done
        self.state[:, self.prev_done] = self.np_random.uniform(
            low=self.low, high=self.high, size=(6, self.prev_done.sum())
        )
        self.steps[self.prev_done] = 0
        reward[self.prev_done] = 0.0
        terminated[self.prev_done] = False
        truncated[self.prev_done] = False
        self.prev_done = np.logical_or(terminated, truncated)

        return self.state.T.astype(np.float32), reward, terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.low, self.high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
        self.state = self.np_random.uniform(
            low=self.low, high=self.high, size=(6, self.num_envs)
        )
        # initial_state = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)  # Modified
        # self.state = np.tile(initial_state, (1, self.num_envs))                                 # Modified

        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_done = np.zeros(self.num_envs, dtype=bool)
        return self.state.T.astype(np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x, _, theta1, _, theta2, _ = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # Cart
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x * scale + self.screen_width / 2.0
        carty = 100
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        # Function to draw a single pole
        def draw_pole(theta, length, color):
            polelen = scale * (2 * length)  # Full length
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole_coords = []
            for coord in [(l, b), (l, t), (r, t), (r, b)]:
                rotated = pygame.math.Vector2(coord).rotate_rad(-theta)
                rotated = (rotated[0] + cartx, rotated[1] + carty + axleoffset)
                pole_coords.append(rotated)
            gfxdraw.aapolygon(self.surf, pole_coords, color)
            gfxdraw.filled_polygon(self.surf, pole_coords, color)

        # Draw both poles with different colors
        draw_pole(theta1, self.length1, (202, 152, 101))  # Brown
        draw_pole(theta2, self.length2, (120, 180, 255))  # Light blue

        # Axle circle
        gfxdraw.aacircle(self.surf, int(cartx), int(carty + axleoffset), int(polewidth / 2), (129, 132, 203))
        gfxdraw.filled_circle(self.surf, int(cartx), int(carty + axleoffset), int(polewidth / 2), (129, 132, 203))

        # Ground line
        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        # Flip vertically
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))


    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
