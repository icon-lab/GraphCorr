import copy


# Turns a dictionary into a class
class Option(object):
      
    def __init__(self, my_dict):

        self.dict = my_dict

        for key in my_dict:
            setattr(self, key, my_dict[key])

    def copy(self):
        return Option(copy.deepcopy(self.dict))


options = Option( {


    
    "expName" : "WILL BE OVERWRITTEN",
    "expGroup" : "WILL BE OVERWRITTEN",


    "workDir" : "/auto/k2/irmak/Projects/GraphCorr_3/", # the absolute path where main.py file resides
    "logRelDir" : "Results",

    # if do not want to init from pretrain original model, then set it to None
    "preTrained_modelRelDir" : None,
    "preTrainFromKFold" : True, # True if want to pretrain on same dataset

    "device" : "cuda:0",

    "supervision" : "supervised",
    "targetTask" : "gender",
    "taskType" : "classification",
    "nOfClasses" : 2,

    "telegramOn" : False,
    "inlineOn" : True,
    "saveOn" : True,
    "tensorboardOn" : True,


    "gamma" : 0.5,
    "stepSize" :20,

    "datasets" : ["hcpRest"], # ["hcpRest", "hcpTask", "abide1", "ppmi"]
    "atlas" : "Schaefer",

    "seed" : 0,
    "schedulerType" : "oneCycle",
    "weightDecay" : 3e-5,
    "dropout" : 0.5,
    
    "lr" : 5e-3,
    "minLr" : 4e-3,
    "maxLr" : 6e-3,

    "nOfEpochs" : 20,
    "kFold" : 5,
    "batchSize" : 24,
    "nominal_batchSize" : 32,
    
    "dynamicLength" : 600, # 600, 100, 
    "windowSize" : 50, # 50, 30
    "stride" : 30, # 20, 10

    "nOfRois" : 400,
    "nOfLayers" : 2,
    "graphDim" : 100, # also used as the hidden dimension of brainNetCnn !!

    "useTransformer": True,
    "useLSTM": False,
    "useTopk": False,
    "nOfParcels": 0,
    "useStatDynFC" : False,

    "poolingMethod" : "gap",
    "brainGnn": Option({
        "poolingRatio" : 0.5, # these are for hcp rest data !!!
        "lamb0" : 1.0,
        "lamb1" : 0.0, # fixed to 0 since they become 0 very fast, see original paper
        "lamb2" : 0.0, # fixed to 0 since they become 0 very fast, see original paper
        "lamb3" : 0.1, # labmda 1
        "lamb4" : 0.1, # lambda 1
        "lamb5" : 0.1, # lambda 2
        "sparsity" : 10


    }),


    "graphLayer" : "sage",

    "useGraphCorr" : False,
    "learnedEmbedDim" : 20,
    "k" : 8,
    "maxLag" : 7,
    "useDistanceMatrix" : True,
    "sparsity_graph" : 2,
    "sparsity_corr" : 2,


    

})



def applyActivation(options_, activation):

    options = options_.copy()
    for key in activation.keys():
        setattr(options, key, activation[key])

    return options
    

def getOptions(activation=None):

    global options

    if(activation != None):
        options = applyActivation(options, activation)

    return options
