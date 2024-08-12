
commonProps = {
    
    "useGraphCorr" : True, # False to ablate GraphCorr
    "sparsity_graph" : 2,
    "sparsity_corr" : 2,
    

    "dynamicLength" : 600,
    "windowSize" : 50,
    "stride" : 30,

    "nOfEpochs" : 20,
    "batchSize" : 12,

    "lr" : 3e-3,
    "minLr" : 2e-3,
    "maxLr" : 4e-3,

    "graphLayer" : "sage",
    "learnedEmbedDim" : 50, #D
    "k" : 3,
    "maxLag" : 5,

    "kFold" : 5,

    "graphDim" : 250,
    "useTransformer": True,
    
    "useDistanceMatrix" : True,
    "useLSTM": False, 

    "datasets" : ["hcpRest"],
    "targetTask" : "gender",

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
