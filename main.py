import torch.nn as nn
import torch.optim as op
import numpy as np
from simulation.runner_speed import run_tests_speed
from simulation.runner_mem import run_tests_mem
from simulation.env import Mem, SqApr, LinApr
from util import save_results

# If the computer that you are running uses Windows please set option WINDOWS to True
WINDOWS = True

# Standard Model for memory
NUMBER_OF_LAYERS_m = 4
NUMBER_OF_NEURONS_m = 50
ACTIVATION_FUNCTION_m = nn.ReLU
DEFAULT_BATCH_SIZE_m = 30
DEFAULT_LEARNING_RATE_m = 0.0001
DEFAULT_OPTIMIZER_m = op.RMSprop

# Memory capacity test
number_of_pattern_tested = tuple(range(1, 302, 10))

number_of_layer_per_neuron = tuple(range(1,21)) # changing number of neurons and number of layers
number_of_neuron_per_layer = tuple(range(5, 51, 5)) # changing number of neurons and number of layers
number_of_layer = tuple(range(1,21))
number_of_neurons = tuple(range(20, 301, 20))  # changing only number of layers

# To perform simulation fill in the variables bellow:
epoch_numb_speed = 100

# Standard Model for speed
NUMBER_OF_LAYERS = 2
NUMBER_OF_NEURONS = 60
ACTIVATION_FUNCTION = nn.ReLU
ARRANGEMENT = lambda x: 1
DEFAULT_BATCH_SIZE = 100
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_OPTIMIZER = op.Adam

# Environments
envs = {"Pattern memorising": Mem,
        "Square function aproximation": SqApr,
        "Linear function aproximation": LinApr}

# 1
optimizer = (
    op.Adadelta, op.Adagrad, op.Adam, op.AdamW, op.Adamax,
    op.ASGD, op.NAdam, op.RAdam, op.RMSprop, op.SGD)

# 2
activation_func = (
    nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU, nn.LogSigmoid,
    nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU, nn.CELU, nn.GELU, nn.Sigmoid, nn.SiLU,
    nn.Mish, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanh, nn.Tanhshrink, nn.Softmax)

# 3
neurons_arrangement = ({"def": lambda x: 1, "str": "Const"},
                       {"def": lambda x: x * 0.3 + 1, "str": "Gęstość 1 + x*0.3"},
                       {"def": lambda x: -x * 0.3 + 1, "str": "Gęstość 1 - x*0.3"},
                       {"def": lambda x: x, "str": "Gęstość x"},
                       {"def": lambda x: -x, "str": "Gęstość -x"})

# 4
number_of_layers = tuple(range(1, 21))

# 5
number_of_neurons_in_layer = tuple(range(10, 100, 5))

# 6
learning_rate = tuple(tuple(np.logspace(-1, -10, num=9, endpoint=False)))

# 7
batch_size = tuple(range(1, 100, 10))

# and start the program
if __name__ == '__main__':
    std = {"opt": DEFAULT_OPTIMIZER,
           "func": ACTIVATION_FUNCTION,
           "arra": ARRANGEMENT,
           "lay": NUMBER_OF_LAYERS,
           "neu": NUMBER_OF_NEURONS,
           "lr": DEFAULT_LEARNING_RATE,
           "bs": DEFAULT_BATCH_SIZE}

    std_m = {"opt": DEFAULT_OPTIMIZER_m,
             "func": ACTIVATION_FUNCTION_m,
             "lay": NUMBER_OF_LAYERS_m,
             "neu": NUMBER_OF_NEURONS_m,
             "lr": DEFAULT_LEARNING_RATE_m,
             "bs": DEFAULT_BATCH_SIZE_m}

    #  Speed test
    tested_configuration_speed = {"opt": optimizer,
                                  "func": activation_func,
                                  "arra": neurons_arrangement,
                                  "lay": number_of_layers,
                                  "neu": number_of_neurons_in_layer,
                                  "lr": learning_rate,
                                  "bs": batch_size,
                                  "std_model": std,
                                  "envs": envs}

    #  Memory test
    tested_configuration_mem = {"problem_size": number_of_pattern_tested,
                                "lay_per_neu": number_of_layer_per_neuron,
                                "neu_per_lay": number_of_neuron_per_layer,
                                "n_lay": number_of_layer,
                                "n_neu": number_of_neurons,
                                "std_model": std_m}

    # Tests

    # res_speed = run_tests_speed(epoch_numb_speed, tested_configuration_speed)
    # save_results(res_speed, WINDOWS)

    res_mem = run_tests_mem(tested_configuration_mem)
    save_results(res_mem, WINDOWS)
