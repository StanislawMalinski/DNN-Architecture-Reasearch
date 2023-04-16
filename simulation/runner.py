from torch.optim import Adam

from simulation.model_builder import ModelBuilder
from simulation.simulation import simulation
import pandas as pd

DEFAULT_BATCH_SIZE = 10
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_OPTIMIZER = Adam


def test_optimizers(epoch, params):
    print("Optimizer simulation:")
    mb = ModelBuilder()
    mb.set_standard()
    label = []
    results = []
    for opt in params:
        print(str(opt))
        res = simulation(opt, mb, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, epoch)
        results.append(res)
        label.append(str(opt))
    r = pd.DataFrame(results, index=label)
    return r


def test_activation_function(epoch, param):
    print("Activation function simulation:")
    mb = ModelBuilder()
    mb.set_standard()
    label = []
    results = []
    for fun in param:
        print(str(fun))
        mb.set_func(fun)
        res = simulation(DEFAULT_OPTIMIZER, mb, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, epoch)
        results.append(res)
        label.append(str(fun))
    r = pd.DataFrame(results, index=label)
    return r


def test_arrangement(epoch, param):
    print("Arrangment simulation:")
    mb = ModelBuilder()
    mb.set_standard()
    label = []
    results = []
    for arra in param:
        mb.set_arrangement(arra["def"])
        res = simulation(DEFAULT_OPTIMIZER, mb, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, epoch)
        results.append(res)
        label.append(arra["str"])
    r = pd.DataFrame(results, index=label)
    return r


def test_layers(epoch, param):
    print("Layer size simulation:")
    mb = ModelBuilder()
    mb.set_standard()
    label = []
    results = []
    for n in param:
        print(f"Layer size {n}")
        mb.set_hidden_layers(n)
        res = simulation(DEFAULT_OPTIMIZER, mb, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, epoch)
        results.append(res)
        label.append("layers=" + str(n))
    r = pd.DataFrame(results, index=label)
    return r


def test_neurons(epoch, param):
    print("Number of neurons simulation:")
    mb = ModelBuilder()
    mb.set_standard()
    label = []
    results = []
    for n in param:
        print(f"Number of neurons {n}")
        mb.set_neurons(n)
        res = simulation(DEFAULT_OPTIMIZER, mb, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, epoch)
        results.append(res)
        label.append("neurons=" + str(n))
    r = pd.DataFrame(results, index=label)
    return r


def test_regularization(epoch, param):
    # TODO
    results = []
    return results


def test_learning_rate(epoch, param):
    print("Learning rate simulation:")
    mb = ModelBuilder()
    mb.set_standard()
    label = []
    results = []
    for lr in param:
        print(f"Learning rate {lr}")
        res = simulation(DEFAULT_OPTIMIZER, mb, DEFAULT_BATCH_SIZE, lr, epoch)
        results.append(res)
        label.append("lr=" + str(lr))
    r = pd.DataFrame(results, index=label)
    return r


def test_batch_size(epoch, param):
    print("Batch size simulation:")
    mb = ModelBuilder()
    mb.set_standard()
    label = []
    results = []
    for bs in param:
        print(f"Batch size {bs}")
        res = simulation(DEFAULT_OPTIMIZER, mb, bs, DEFAULT_LEARNING_RATE, epoch)
        results.append(res)
        label.append("bs=" + str(bs))
    r = pd.DataFrame(results, index=label)
    return r


def run_tests(epoch: int, configuration: dict):
    res = dict()
    if "opt" in configuration.keys():
        r = test_optimizers(epoch, configuration["opt"])
        res["opt"] = r
    '''
    if "func" in configuration.keys():
        r = test_activation_function(epoch, configuration["func"])
        res["func"] = r

    if "arra" in configuration.keys():
        r = test_arrangement(epoch, configuration["arra"])
        res["arra"] = r

    if "lay" in configuration.keys():
        r = test_layers(epoch, configuration["lay"])
        res["lay"] = r

    if "neu" in configuration.keys():
        r = test_neurons(epoch, configuration["neu"])
        res["neu"] = r

    if "reg" in configuration.keys():
        r = test_regularization(epoch, configuration["reg"])
        res["reg"] = r

    if "lr" in configuration.keys():
        r = test_learning_rate(epoch, configuration["lr"])
        res["lr"] = r

    if "bs" in configuration.keys():
        r = test_batch_size(epoch, configuration["bs"])
        res["bs"] = r
    '''
    return res
