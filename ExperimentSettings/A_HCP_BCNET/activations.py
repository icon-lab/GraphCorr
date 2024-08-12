
commonProps = {
    
    "windowSize" : 50,
    "stride" : 30,


    "useGraphCorr" : True,
    "sparsity_graph" : 2,
    "sparsity_corr" : 2,

    "weightDecay" : 5e-4,
    "dropout" : 0.1,
    "lr" : 2e-4,
    "minLr" : 2e-5,
    "maxLr" : 4e-4,
    "graphDim" : 32,
    "dynamicLength" : 600, 
    "batchSize" : 16,

    "datasets" : ["hcpRest"],
    "targetTask" : "gender",
    "atlas" : "AAL",
    "nOfRois" : 116,

    "learnedEmbedDim" : 50, #D
    "k" : 3,
    "maxLag" : 5,
    "graphLayer" : "brainNetCnn",
    "telegramOn" : False,
    "seed" : 0
}

activations = [
    {"expId" : 1}
    
]

temp = []

for i, activation in enumerate(activations):
    
    baseProps = commonProps.copy()

    for prop in activation:
        baseProps[prop] = activation[prop]
    
    temp.append(baseProps)

activations = temp

