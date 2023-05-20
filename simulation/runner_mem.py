import pandas as pd

from simulation.model_builder import ModelBuilder
from simulation.simulation import simulation_mem
from simulation.status import Status

# tested_configuration_mem = {"problem_size": number_of_pattern_tested,
#                            "neu_per_lay": number_of_neuron_per_layer,
#                            "n_lay": number_of_layer,
#                            "n_neu": number_of_neurons,
#                            "std_model": std}

def test_neuron_per_layer(std, mem_size ,configurations):
    mb = ModelBuilder(std)
    mb.clear()
    label = []
    results = []
    Status.get_status().set_param("Neurons per layer")
    for configuration in configurations:
        Status.get_status().set_value(configuration)
        mb.set_neurons(mb.get_hidden_layers() * configuration)
        res = simulation_mem(mb, mem_size, std["opt"], std["lr"], std["bs"])
        results.append(res)
        label.append("neu_per_lay=" + str(configuration))
    r = pd.DataFrame(results, index=label, columns=mem_size)
    return r.T


def test_number_of_layers(std, mem_size, configurations):
    mb = ModelBuilder(std)
    mb.clear()
    label = []
    results = []
    Status.get_status().set_param("Number of layers")
    for configuration in configurations:
        Status.get_status().set_value(configuration)
        mb.set_hidden_layers(configuration)
        res = simulation_mem(mb,mem_size, std["opt"], std["lr"], std["bs"])
        results.append(res)
        label.append("n_lay=" + str(configuration))
    r = pd.DataFrame(results, index=label, columns=mem_size)
    return r.T


def test_number_of_neurons(std,mem_size,  configurations):
    mb = ModelBuilder(std)
    mb.clear()
    label = []
    results = []
    Status.get_status().set_param("Number of neurons")
    for configuration in configurations:
        Status.get_status().set_value(configuration)
        mb.set_neurons(configuration)
        res = simulation_mem(mb,mem_size, std["opt"], std["lr"], std["bs"])
        results.append(res)
        label.append("n_neu=" + str(configuration))
    r = pd.DataFrame(results, index=label, columns=mem_size)
    return r.T


def test_layer_per_neurons(std,mem_size,  configurations):
    mb = ModelBuilder(std)
    mb.clear()
    label = []
    results = []
    Status.get_status().set_param("Layers per neurons")

    for configuration in configurations:
        Status.get_status().set_value(configuration)
        mb.set_neurons(mb.get_neurons() * configuration)
        res = simulation_mem(mb,mem_size, std["opt"], std["lr"], std["bs"])
        results.append(res)
        label.append("lay_per_neu=" + str(configuration))
    r = pd.DataFrame(results, index=label, columns=mem_size)
    return r.T


def init_status_bar(configuration):
    x_tics = len(configuration["problem_size"])
    per = 0
    for key in configuration.keys():
        if key == "problem_size" or key == "std_model":
            continue
        per += len(configuration[key])
    Status(per * 20 * x_tics)


def run_tests_mem(configuration: dict):
    init_status_bar(configuration)
    Status.get_status().set_env("Memory")
    res = dict()
    std = configuration["std_model"]

    std["arra"] = lambda x: 1

    if "neu_per_lay" in configuration.keys():
        r = test_neuron_per_layer(std, configuration["problem_size"], configuration["neu_per_lay"])
        res["neu_per_lay"] = r

    if "lay_per_neu" in configuration.keys():
        r = test_neuron_per_layer(std, configuration["problem_size"], configuration["lay_per_neu"])
        res["lay_per_neu"] = r

    if "n_lay" in configuration.keys():
        r = test_number_of_layers(std, configuration["problem_size"], configuration["n_lay"])
        res["n_lay"] = r

    if "n_neu" in configuration.keys():
        r = test_number_of_neurons(std, configuration["problem_size"], configuration["n_neu"])
        res["n_neu"] = r
    return {"Memory test": res}
