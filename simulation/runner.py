from simulation.model_builder import ModelBuilder
from simulation.simulation import simulation
import pandas as pd


def test_optimizers(epoch, params, std_model):
    print("Optimizer simulation:")
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    for opt in params:
        print(str(opt))
        res = simulation(mb, opt, std_model["bs"], std_model["lr"], epoch)
        results.append(res)
        label.append(str(opt))
    r = pd.DataFrame(results, index=label)
    return r.T


def test_activation_function(epoch, param, std_model):
    print("Activation function simulation:")
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    for fun in param:
        print(str(fun))
        mb.set_func(fun)
        res = simulation(mb, std_model["opt"], std_model["bs"], std_model["lr"], epoch)
        results.append(res)
        label.append(str(fun))
    r = pd.DataFrame(results, index=label)
    return r.T


def test_arrangement(epoch, param, std_model):
    print("Arrangment simulation:")
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    for arra in param:
        mb.set_arrangement(arra["def"])
        res = simulation(mb, std_model["opt"], std_model["bs"], std_model["lr"], epoch)
        results.append(res)
        label.append(arra["str"])
    r = pd.DataFrame(results, index=label)
    return r.T


def test_layers(epoch, param, std_model):
    print("Layer size simulation:")
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    for n in param:
        print(f"Layer size {n}")
        mb.set_hidden_layers(n)
        res = simulation(mb, std_model["opt"], mb, std_model["bs"], std_model["lr"], epoch)
        results.append(res)
        label.append("layers=" + str(n))
    r = pd.DataFrame(results, index=label)
    return r.T


def test_neurons(epoch, param, std_model):
    print("Number of neurons simulation:")
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    for n in param:
        print(f"Number of neurons {n}")
        mb.set_neurons(n)
        res = simulation(mb, std_model["opt"], std_model["bs"], std_model["lr"], epoch)
        results.append(res)
        label.append("neurons=" + str(n))
    r = pd.DataFrame(results, index=label)
    return r.T


def test_learning_rate(epoch, param, std_model):
    print("Learning rate simulation:")
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    for lr in param:
        print(f"Learning rate {lr}")
        res = simulation(mb, std_model["opt"], std_model["bs"], lr, epoch)
        results.append(res)
        label.append("lr=" + str(lr))
    r = pd.DataFrame(results, index=label)
    return r.T


def test_batch_size(epoch, param, std_model):
    print("Batch size simulation:")
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    for bs in param:
        print(f"Batch size {bs}")
        res = simulation(mb, std_model["opt"], bs, std_model["lr"], epoch)
        results.append(res)
        label.append("bs=" + str(bs))
    r = pd.DataFrame(results, index=label)
    return r.T


def run_tests(epoch: int, configuration: dict):
    res = dict()
    std = configuration["std_model"]
    if "opt" in configuration.keys():
        r = test_optimizers(epoch, configuration["opt"], std)
        res["opt"] = r

    if "func" in configuration.keys():
        r = test_activation_function(epoch, configuration["func"], std)
        res["func"] = r

    if "arra" in configuration.keys():
        r = test_arrangement(epoch, configuration["arra"], std)
        res["arra"] = r

    if "lay" in configuration.keys():
        r = test_layers(epoch, configuration["lay"], std)
        res["lay"] = r

    if "neu" in configuration.keys():
        r = test_neurons(epoch, configuration["neu"], std)
        res["neu"] = r

    if "lr" in configuration.keys():
        r = test_learning_rate(epoch, configuration["lr"], std)
        res["lr"] = r

    if "bs" in configuration.keys():
        r = test_batch_size(epoch, configuration["bs"], std)
        res["bs"] = r

    return res
