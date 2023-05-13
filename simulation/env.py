import random
from abc import ABC, abstractmethod

import torch
from simulation.model_builder import CUDA


class Env(ABC):
    def __init__(self):
        self._required_sizes = None
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

    def get_required_sizes(self):
        return self._required_sizes

    @abstractmethod
    def observation(self):
        return None

    @abstractmethod
    def reset(self):
        return None  # resets

    @abstractmethod
    def step(self, action):
        return None  # observation, error


class Mem(Env):

    def __init__(self, pattern=20):
        super().__init__()
        self.number_of_patterns = pattern

    def reset(self):
        self._check_sizes()
        self.set = []
        torch.manual_seed(1)
        for i in range(self.number_of_patterns):
            A = torch.round(torch.rand(self._input_size))
            B = torch.round(torch.rand(self._output_size))

            self.set.append((A, B))
        return self.observation()

    def observation(self):
        self.__state = random.sample(self.set, 1)[0]
        if CUDA:
            return self.__state[0].to(torch.device('cuda:0'))
        return self.__state[0]

    def step(self, action):
        y = self.__state[1]
        if CUDA:
            return self.observation(), y.to(torch.device('cuda:0'))
        return self.observation(), y


class SqApr(Env):
    def reset(self):
        self._check_sizes()
        self.weight


    def observation(self):
        self.__state = torch.rand(self._input_size)
        if CUDA:
            return self.__state.to(torch.device('cuda:0'))
        return self.__state

    def step(self, action):
        y = (self.__state ** 2).sum() / self.get_output_size()
        if CUDA:
            return self.observation(), torch.tensor([y] * self.get_output_size()).to(torch.device('cuda:0'))
        return self.observation(), torch.tensor([y] * self.get_output_size())


class LinApr(Env):
    def reset(self):
        self._check_sizes()
        A = torch.rand(self._input_size, self.get_output_size())
        B = torch.rand(1, self.get_output_size())
        C = [A.sum(dim=0) + B for _ in range(self.get_input_size())]
        sum = torch.cat(C, 0)
        self.A = A.div(sum)
        self.B = B.div(sum)

    def observation(self):
        self.__state = torch.rand(self._input_size)
        if CUDA:
            return self.__state.to(torch.device('cuda:0'))
        return self.__state

    def step(self, action):
        y = self.A * action + self.B # is Action size (4, 1)?
        if CUDA:
            return self.observation(), torch.tensor([y] * self.get_output_size()).to(torch.device('cuda:0'))
        return self.observation(), y


A = torch.rand(20, 20)
B = torch.rand(1, 20)
Asum = A.sum(dim=1) + B