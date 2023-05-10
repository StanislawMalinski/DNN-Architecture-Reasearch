from simulation.model_builder import ModelBuilder
from simulation.simulation import simulation
import pandas as pd
import re

from simulation.status import Status


def test_optimizers(std_model, env ,params, epoch):
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    Status.get_status().set_param("optimizer")
    for opt in params:
        name = re.search(r'\w+\'',str(opt)).group()[0:-1]
        Status.get_status().set_value(name)
        res = simulation(mb, env(), opt, std_model["bs"], std_model["lr"], epoch)
        results.append(res)
        label.append(name)
    r = pd.DataFrame(results, index=label)
    return r.T


def test_activation_function(std_model, env ,params, epoch):
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    Status.get_status().set_param("activation function")
    for fun in params:
        name = re.search(r'\w+\'', str(fun)).group()[0:-1]
        Status.get_status().set_value(name)
        mb.set_func(fun)
        res = simulation(mb, env(), std_model["opt"], std_model["bs"], std_model["lr"], epoch)
        results.append(res)
        label.append(name)
    r = pd.DataFrame(results, index=label)
    return r.T


def test_arrangement(std_model, env ,params, epoch):
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    Status.get_status().set_param("arrangment")
    for arra in params:
        Status.get_status().set_value(arra["str"])
        mb.set_arrangement(arra["def"])
        res = simulation(mb, env(), std_model["opt"], std_model["bs"], std_model["lr"], epoch)
        results.append(res)
        label.append(arra["str"])
    r = pd.DataFrame(results, index=label)
    return r.T


def test_layers(std_model, env ,params, epoch):
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    Status.get_status().set_param("number of hidden layers")
    for n in params:
        Status.get_status().set_value(n)
        mb.set_hidden_layers(n)
        res = simulation(mb, env(), std_model["opt"], std_model["bs"], std_model["lr"], epoch)
        results.append(res)
        label.append("layers=" + str(n))
    r = pd.DataFrame(results, index=label)
    return r.T


def test_neurons(std_model, env ,params, epoch):
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    Status.get_status().set_param("neurons")
    for n in params:
        Status.get_status().set_value(n)
        mb.set_neurons(n)
        res = simulation(mb, env(), std_model["opt"], std_model["bs"], std_model["lr"], epoch)
        results.append(res)
        label.append("neurons=" + str(n))
    r = pd.DataFrame(results, index=label)
    return r.T


def test_learning_rate(std_model, env ,params, epoch):
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    Status.get_status().set_param("learning rate")
    for lr in params:
        Status.get_status().set_value(lr)
        res = simulation(mb, env(), std_model["opt"], std_model["bs"], lr, epoch)
        results.append(res)
        label.append("lr=" + str(lr))
    r = pd.DataFrame(results, index=label)
    return r.T


def test_batch_size(std_model, env ,params, epoch):
    mb = ModelBuilder(std_model)
    mb.clear()
    label = []
    results = []
    Status.get_status().set_param("batch")
    for bs in params:
        Status.get_status().set_value(bs)
        res = simulation(mb, env(), std_model["opt"], bs, std_model["lr"], epoch)
        results.append(res)
        label.append("bs=" + str(bs))
    r = pd.DataFrame(results, index=label)
    return r.T

def init_status_bar(epoch, configuration):
    envs = len(configuration["envs"].keys())
    per_env = 0
    for key in configuration.keys():
        if key == "envs" or key == "std_model":
            continue
        per_env += len(configuration[key])
    Status(epoch * envs * per_env)

def run_tests_speed(epoch: int, configuration: dict):
    keys = configuration["envs"].keys()
    envs = configuration["envs"]
    init_status_bar(epoch, configuration)
    results = dict()
    for key in keys:
        env = envs[key]
        Status.get_status().set_env(key)
        res = dict()
        std = configuration["std_model"]
        if "opt" in configuration.keys():
            r = test_optimizers(std, env, configuration["opt"], epoch)
            res["opt"] = r

        if "func" in configuration.keys():
            r = test_activation_function(std, env, configuration["func"], epoch)
            res["func"] = r

        if "arra" in configuration.keys():
            r = test_arrangement(std, env, configuration["arra"], epoch)
            res["arra"] = r

        if "lay" in configuration.keys():
            r = test_layers(std, env, configuration["lay"], epoch)
            res["lay"] = r

        if "neu" in configuration.keys():
            r = test_neurons(std, env, configuration["neu"], epoch)
            res["neu"] = r

        if "lr" in configuration.keys():
            r = test_learning_rate(std, env, configuration["lr"], epoch)
            res["lr"] = r

        if "bs" in configuration.keys():
            r = test_batch_size(std, env, configuration["bs"], epoch)
            res["bs"] = r

        results[key] = res

    return results
