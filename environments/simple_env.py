import gymnasium as gym
import numpy as np

from gymnasium import spaces


class Simple2DNavigationEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # State bounds
        self.state_lower_bound = np.array([-1., -1., -1., -1.], dtype=np.float32)  # Limit velocity bounds
        self.state_upper_bound = np.array([1., 1., 1., 1.], dtype=np.float32)      # Limit velocity bounds

        # Action bounds
        self.action_lower_bound = -0.1
        self.action_upper_bound = 0.1

        # Spaces
        self.action_space = spaces.Box(
            low=self.action_lower_bound,
            high=self.action_upper_bound,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=self.state_lower_bound,
            high=self.state_upper_bound,
            dtype=np.float32,
        )

        self.dt = 1
        self.max_steps = 5000
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize state: random position, zero velocity
        self.state = np.zeros(4, dtype=np.float32)
        self.state[:2] = self.np_random.uniform(
            self.state_lower_bound[:2],
            self.state_upper_bound[:2]
        ).astype(np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.steps += 1
        action = np.array(action, dtype=np.float32)
        position = self.state[:2].copy()
        velocity = self.state[2:].copy()

        # Update state
        position += velocity * self.dt + 0.5 * action * self.dt**2
        velocity += action * self.dt

        # Handle boundaries
        for i in range(2):
            if position[i] < self.state_lower_bound[i]:
                velocity[i] = -0.1 * velocity[i]
                position[i] = self.state_lower_bound[i]
            elif position[i] > self.state_upper_bound[i]:
                velocity[i] = -0.1 * velocity[i]
                position[i] = self.state_upper_bound[i]

        # Clip velocity to observation space bounds
        velocity = np.clip(velocity, self.state_lower_bound[2:], self.state_upper_bound[2:])

        self.state = np.concatenate([position, velocity]).astype(np.float32)

        # Check termination
        done = (np.allclose(position, 0, atol=0.25) and
                np.allclose(velocity, 0, atol=0.1)) or self.steps >= self.max_steps

        return self.state, np.float32(-0.01), done, False, {}

# Register the environment
gym.register(
    id='Simple2DNavigation-v0',
    entry_point='environments.simple_env:Simple2DNavigationEnv',
)
