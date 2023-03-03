import torch.nn as nn
import torch.optim as op

from simulation import run_tests

# To perform simulation fill in the variables bellow:
batch_size = 30
epoch_numb = 100

activation_func = (
    nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU, nn.LogSigmoid,
    nn.MultiheadAttention, nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU, nn.CELU, nn.GELU, nn.Sigmoid,
    nn.SiLU, nn.Mish, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanh, nn.Tanhshrink, nn.Threshold, nn.GLU,
    nn.Softmin, nn.Softmax, nn.Softmax2d, nn.LogSoftmax, nn.AdaptiveLogSoftmaxWithLoss)

number_of_layers = tuple(range(1, 21))

number_of_neurons_in_layer = tuple(range(10, 50, 5))

optimizer = (
    op.Adadelta, op.Adagrad, op.Adam, op.AdamW, op.SparseAdam, op.Adamax, op.ASGD, op.LBFGS, op.NAdam, op.RAdam,
    op.RMSprop, op.Rprop)

# and start the program
if __name__ == '__main__':
    tested_configuration = {"func": activation_func, "lay": number_of_layers, "neu": number_of_neurons_in_layer,
                            "opt": optimizer}
    run_tests(batch_size, epoch_numb, tested_configuration)
