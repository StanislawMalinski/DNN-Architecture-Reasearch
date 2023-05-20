import torch
import torch.nn as nn
from math import floor

CUDA = True

class Model(nn.Module):
    def __init__(self, seq):
        super(Model, self).__init__()
        self.pipe = nn.Sequential(*seq)

    def forward(self, x):
        return self.pipe(x)


class ModelBuilder:
    def __init__(self, std):
        self.std = std
        self.clear()

    def set_input(self, n):
        self.num_input = n

    def set_classes(self, n):
        self.num_classes = n

    def set_hidden_layers(self, n):
        self.hiden_layer = n

    def get_hidden_layers(self):
        return self.hiden_layer

    def set_arrangement(self, arrangement):
        self.arrangement = arrangement

    def set_func(self, func):
        self.func = func

    def set_neurons(self, n):
        self.num_of_neurons = n

    def get_neurons(self):
        return self.num_of_neurons

    def clear(self):
        self.func = self.std["func"]
        self.hiden_layer = self.std["lay"]
        self.arrangement = self.std["arra"]
        self.num_of_neurons = self.std["neu"]
        self.num_input = -1
        self.num_classes = -1

    def __get_proportion(self):
        sum = 0
        assert self.hiden_layer > 0
        prop = [0] * self.hiden_layer
        for lay in range(self.hiden_layer):
            sum += self.arrangement(lay)
            prop[lay] = self.arrangement(lay)

        n = self.num_of_neurons

        for lay in range(self.hiden_layer - 1):
            if 10 >= (prop[lay] / sum) * self.num_of_neurons:
                Exception("Insufficient number of neurons per layer.")
            prop[lay] = floor((prop[lay] / sum) * self.num_of_neurons)
            n -= prop[lay]

        prop[-1] = n
        prop += [self.num_classes]
        return [self.num_input] + prop

    def __check_io_layers(self):
        if self.num_classes == -1 or self.num_input == -1:
            Exception("Size of input or output layer hasn't been set.")

    def finalize(self):
        self.__check_io_layers()
        arrang = self.__get_proportion()
        seq = []
        for l in range(self.hiden_layer + 1):
            if self.func is not None:
                seq.append(self.func())
            lay = nn.Linear(arrang[l], arrang[l + 1])
            w = lay.weight.clone()
            b = lay.bias.clone()

            w[:] = 0.00001
            b[:] = 0.00001
            lay.weight = nn.Parameter(w)
            lay.bias = nn.Parameter(b)

            seq.append(lay)
        net = Model(seq)
        if CUDA:
            net.to(torch.device('cuda:0'))
        return net

