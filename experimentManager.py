from run import experimentRunner
import importlib
from option import getOptions
from termcolor import colored
import traceback
import torch


def experimenter(experimentGroup, reRun):


    activations = importlib.import_module("ExperimentSettings." + experimentGroup + ".activations").activations

    print("Total number of activations = {}".format(len(activations)))

    failedRuns = []

    for i, activation in enumerate(activations):

        options = getOptions(activation)

        torch.manual_seed(options.seed)
        
        gpu = 0

        options.device = "cuda:" + str(gpu)
        options.expGroup = experimentGroup
        options.expName = "activation{}".format(options.expId)

        try:
            print(colored("Experiment Group {} - Running activation : {} on gpu : {}".format(experimentGroup, i, gpu), "green"))            
                    
            experimentRunner(options, reRun)

        except Exception as e:

            print("\n\n{}\n".format(traceback.format_exc()))

            print(colored("Experiment Group {} - {}'th activation failed".format(experimentGroup, i), "red"))
            failedRuns.append(i)
            continue

        torch.cuda.empty_cache()


        print(colored("Experiment Group {} - Finished activation : {}".format(experimentGroup, i), "green"))

    print(colored("Experiment Group {} finished - {} activations failed".format(experimentGroup, failedRuns), "blue"))
    