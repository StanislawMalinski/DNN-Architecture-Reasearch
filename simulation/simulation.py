from simulation_arrangement import test_arrangement
from simulation_batch_size import test_batch_size
from simulation_learning_rate import test_learning_rate
from simulation_regularization import test_regularization
from simulation_layer import test_layers
from simulation_func import test_activation_function
from simulation_neurons import test_neurons
from simulation_optimizers import test_optimizers

def run_tests(batch: int, epoch: int, configuration: dict):
    res = dict()
    if "opt" in configuration.keys():
        r = test_optimizers(batch, epoch, configuration["opt"])
        res["opt"] = r

    if "func" in configuration.keys:
        r = test_activation_function(batch, epoch, configuration["func"])
        res["func"] = r

    if "arra" in configuration.keys():
        r = test_arrangement(batch, epoch, configuration["arra"])
        res["arra"] = r

    if "lay" in configuration.keys():
        r = test_layers(batch, epoch, configuration["lay"])
        res["lay"] = r

    if "neu" in configuration.keys():
        r = test_neurons(batch, epoch, configuration["neu"])
        res["neu"] = r

    if "reg" in configuration.keys():
        r = test_regularization(batch, epoch, configuration["reg"])
        res["reg"] = r

    if "lr" in configuration.keys():
        r = test_learning_rate(batch, epoch, configuration["lr"])
        res["lr"] = r

    if "bs" in configuration.keys():
        r = test_batch_size(batch, epoch, configuration["bs"])
        res["bs"] = r
    return res



