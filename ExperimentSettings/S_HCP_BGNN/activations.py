
commonProps = {
    
    "windowSize" : 50,
    "stride" : 30,

    "batchSize" : 12,
    "weightDecay" : 5e-3,
    "nOfEpochs" : 50,

    "lr" : 1e-2,
    "gamma" : 0.5,
    "stepSize" :20,

    "datasets" : ["hcpRest"],
    "targetTask" : "gender",

    "useGraphCorr" : True,
    "sparsity_graph" : 2,
    "sparsity_corr" : 2,

    "dynamicLength" : 600, 

    "device" : "cuda:0",
    "learnedEmbedDim" : 50, #D
    "k" : 3,
    "maxLag" : 5,
    "graphLayer" : "brainGnn",
    "telegramOn" : False,
    "seed" : 0
}

activations = [
    {"expId" : 0}

]

temp = []

for i, activation in enumerate(activations):
    
    baseProps = commonProps.copy()

    for prop in activation:
        baseProps[prop] = activation[prop]
    
    temp.append(baseProps)

activations = temp
