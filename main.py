import torch.nn as nn
import torch.optim as op
import numpy as np

# To perform simulation fill in the variables bellow:
epoch_numb = 100

# 1
optimizer = (
    op.Adadelta, op.Adagrad, op.Adam, op.AdamW, op.SparseAdam, op.Adamax, op.ASGD, op.LBFGS, op.NAdam, op.RAdam,
    op.RMSprop, op.Rprop)

# 2
activation_func = (
    nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU, nn.LogSigmoid,
    nn.MultiheadAttention, nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU, nn.CELU, nn.GELU, nn.Sigmoid,
    nn.SiLU, nn.Mish, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanh, nn.Tanhshrink, nn.Threshold, nn.GLU,
    nn.Softmin, nn.Softmax, nn.Softmax2d, nn.LogSoftmax, nn.AdaptiveLogSoftmaxWithLoss)

#3
neurons_arrangement = ()

#4
number_of_layers = tuple(range(1, 21))

#5
number_of_neurons_in_layer = tuple(range(10, 50, 5))

#6
regularization_techniques = ()

#7
learning_rate = tuple(tuple(np.logspace(0, -15, num=30, endpoint=False)))

#8
batch_size = tuple(range(1, 10001, 100))

# and start the program
if __name__ == '__main__':
    tested_configuration = {"opt": optimizer,
                            "func": activation_func,
                            "arran": neurons_arrangement,
                            "lay": number_of_layers,
                            "neu": number_of_neurons_in_layer,
                            "reg": regularization_techniques,
                            "lr": learning_rate,
                            "bs": batch_size}
    run_tests(epoch_numb, tested_configuration)
