import random
from abc import ABC, abstractmethod

import torch

class Env(ABC):
    def __init__(self):
        self._input_size = -1
        self._output_size = -1

    def _check_sizes(self):
        if self._input_size == -1:
            Exception("Haven't set input size layer.")
        if self._output_size == -1:
            Exception("Haven't set output size layer.")

    def set_input_size(self, x):
        self._input_size = x

    def set_output_size(self, x):
        self._output_size = x

    def get_input_size(self):
        return self._input_size

    def get_output_size(self):
        return self._output_size

    @abstractmethod
    def observation(self):
        return None

    @abstractmethod
    def reset(self):
        return None  # resets

    @abstractmethod
    def step(self, action):
        return None  # observation, error

NUMBER_OF_PATTERNS = 20

class Mem(Env):
    def reset(self):
        self.set = []
        torch.manual_seed(0)
        for i in range(NUMBER_OF_PATTERNS):
            A = torch.rand(self._input_size)
            B = torch.rand(self.get_output_size())

            self.set.append((A, B))
        return self.observation()

    def observation(self):
        self._check_sizes()
        self.__state = random.sample(self.set, 1)[0]
        return self.__state[0]

    def step(self, action):
        y = self.__state[1]
        return self.observation(), y



