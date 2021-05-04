"""Agent for a Walker2DBullet environment."""
from os.path import abspath, dirname, realpath

from gym.spaces.box import Box
import numpy as np


ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory


class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self):
        self._action_space = Box(-1, 1, (6,))
        self._action_space.seed(0)

    def act(self, observation: np.ndarray):
        """Decides which action to take for the given observation."""
        del observation
        return self._action_space.sample()

    def load(self):
        """Loads network parameters if there are any.

        Example:
            path = join(ROOT, 'model.pth')
            self.policy.load_state_dict(torch.load(path))
        """
