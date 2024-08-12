import numpy as np
import copy

### you need a fully trained model before saliency analysis

class Option(object):
      
    def __init__(self, my_dict):

        self.dict = my_dict

        for key in my_dict:
            setattr(self, key, my_dict[key])

    def copy(self):
        return Option(copy.deepcopy(self.dict))

##### sage hcp example
options1 = Option({"expName" : "activation0",  
"saveName" : "sage_hcp", # save name for analysis outputs
"expGroup" : None, # experiment directory name
"workDir" : None, # your directory for the project
"logRelDir" : "Results", 
"device" : "cuda:0", 
"supervision" : "supervised",
"targetTask" : "gender",
"datasets" : ['hcpRest'],
"atlas" : "Schaefer",
"seed" : 0,
"kFold" : 5,
"batchSize" : 1,
"dynamicLength" : 1200,
"windowSize" : 50,
"stride" : 30,
"graphLayer" : "sage",
"useGraphCorr" : False, # or True based on whşch experiment you are doing
"useDistanceMatrix" : True,
"sparsity_graph" : 2,
"sparsity_corr" : 2,
"expId" : None}) # your experiment ID


def getOptions(exp):

    if(exp == "sage_hcp"):
        options = options1
    # fill this up with any more experiments of your choice
    else:
        options = None

    return options