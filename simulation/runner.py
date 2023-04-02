from torch.optim import Adam

from simulation.model_builder import ModelBuilder

DEFAULT_BATCH_SIZE = 100
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_OPTIMIZER = Adam


def test_optimizers(epoch, params):
    mb = ModelBuilder()
    mb.set_standard()
    results = []
    for opt in params:
        res = simulation(opt, mb, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, epoch)
        results += (res, str(opt))
    return results


def test_activation_function(epoch, param):
    mb = ModelBuilder()
    mb.set_standard()
    results = []
    for fun in param:
        mb.set_func(fun)
        res = simulation(DEFAULT_OPTIMIZER, mb, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, epoch)
        results += (res, str(fun))
    return results


def test_arrangement(epoch, param):
    mb = ModelBuilder()
    mb.set_standard()
    results = []
    for arra in param:
        mb.set_arrangement(arra["def"])
        res = simulation(DEFAULT_OPTIMIZER, mb, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, epoch)
        results += (res, arra["str"])
    return results


def test_layers(epoch, param):
    mb = ModelBuilder()
    mb.set_standard()
    results = []
    for n in param:
        mb.set_hidden_layers(n)
        res = simulation(DEFAULT_OPTIMIZER, mb, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, epoch)
        results += (res, "layers=" + n)
    return results


def test_neurons(epoch, param):
    mb = ModelBuilder()
    mb.set_standard()
    results = []
    for n in param:
        mb.set_neurons(n)
        res = simulation(DEFAULT_OPTIMIZER, mb, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, epoch)
        results += (res, "neurons=" + n)
    return results


def test_regularization(epoch, param):
    # TODO
    results = []
    return results


def test_learning_rate(epoch, param):
    mb = ModelBuilder()
    mb.set_standard()
    results = []
    for lr in param:
        res = simulation(DEFAULT_OPTIMIZER, mb, DEFAULT_BATCH_SIZE, lr, epoch)
        results += (res, "lr=" + lr)
    return results


def test_batch_size(epoch, param):
    mb = ModelBuilder()
    mb.set_standard()
    results = []
    for bs in param:
        res = simulation(DEFAULT_OPTIMIZER, mb, bs, DEFAULT_LEARNING_RATE, epoch)
        results += (res, "bs=" + bs)
    return results


def run_tests(epoch: int, configuration: dict):
    res = dict()
    if "opt" in configuration.keys():
        r = test_optimizers(epoch, configuration["opt"])
        res["opt"] = r

    if "func" in configuration.keys:
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
    return res
