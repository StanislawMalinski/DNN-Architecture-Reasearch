from simulation_layer import test_layers
from simulation_func import test_activation_function
from simulation_neurons import test_neurons
from simulation_optimizers import test_optimizers

def run_tests(batch: int, epoch: int, configuration: dict):
    res = dict()
    if "func" in configuration.keys:
        r = test_activation_function(batch, epoch, configuration["func"])
        res["func"] = r

    if "lay" in configuration.keys():
        r = test_layers(batch, epoch, configuration["lay"])
        res["lay"] = r

    if "neu" in configuration.keys():
        r = test_neurons(batch, epoch, configuration["neu"])
        res["neu"] = r

    if "opt" in configuration.keys():
        r = test_optimizers(batch, epoch, configuration["opt"])
        res["opt"] = r

    if "arch" in configuration.keys():
        res["func"] = r
        pass

    return res



