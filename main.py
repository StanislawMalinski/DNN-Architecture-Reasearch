import torch.nn as nn
import torch.optim as op
import numpy as np
import os
import datetime

import visualisation
from simulation.runner import run_tests

# To perform simulation fill in the variables bellow:
epoch_numb = 30

# Standard Model
NUMBER_OF_LAYERS = 2
NUMBER_OF_NEURONS = 60
ACTIVATION_FUNCTION = nn.ReLU
ARRANGEMENT = lambda x: 1
DEFAULT_BATCH_SIZE = 100
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_OPTIMIZER = op.Adam

# 1
optimizer = (
    op.Adadelta, op.Adagrad, op.Adam, op.AdamW, op.Adamax,
    op.ASGD, op.NAdam, op.RAdam, op.RMSprop, op.Rprop, op.SGD)

# 2
activation_func = (
    nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU, nn.LogSigmoid,
    nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU, nn.CELU, nn.GELU, nn.Sigmoid, nn.SiLU,
    nn.Mish, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanh, nn.Tanhshrink, nn.Softmax)

# 3
neurons_arrangement = ({"def": lambda x: 1, "str": "Const"},
                       {"def": lambda x: x * 0.3 + 1, "str": "Liniowy od 1"},
                       {"def": lambda x: -x * 0.3 + 1, "str": "Liniowy do 1"})

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

    tested_configuration = {"opt": optimizer,
                            "func": activation_func,
                            "arra": neurons_arrangement,
                            "lay": number_of_layers,
                            "neu": number_of_neurons_in_layer,
                            "lr": learning_rate,
                            "bs": batch_size,
                            "std_model": std}

    # res = run_tests(epoch_numb, tested_configuration)

    res = run_tests(epoch_numb, {"lr": learning_rate,"std_model": std})

    df = res["lr"]
    visualisation.visualise(df)

    date = datetime.datetime.now().strftime("%x").replace("/", "-")
    dir = f"C:\\Users\\Staszek\\Documents\\Investing\\DNNReasearch\\Results\\Results({date})"

    visualisation.visualise(res["lr"])

    if not os.path.isdir(dir):
        os.mkdir(dir)

    for key in res.keys():
        df = res[key]
        df.to_csv(f"C:\\Users\\Staszek\\Documents\\Investing\\DNNReasearch\\Results\\Results({date})\\{key}.csv")
