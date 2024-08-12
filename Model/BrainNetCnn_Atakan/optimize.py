import wandb
import torch
import time
import os
import sys
from run import run_brainNetCnn

optimizerTemp = torch.load("./optimizerTemp.save")
device = optimizerTemp["device"]
datasetSeeds = optimizerTemp["datasetSeeds"]

sys.path.append("../../../")
from Dataset.datasetDetails import datasetDetailsDict

from utils import Option, metricSummer, calculateMetrics


#########################
# HERE DEFINE DEFAULT PARAMETERS
# THESE ARE MEANT TO GIVE AN IDEA WHAT SHOULD BE THE VALUES TO OPTMIZE, AND POSSIBLY ABOUT THEIR RANGE

hyperParams = dict(

    batchSize = 16, # categ : [8, 16, 32]
    weightDecay = 3e-5, # log uniform : max: 5e-3, min: 5e-7
    nOfEpochs = 15, # categ : [5, 10, 15, 20]

    lr = 5e-4, # log uniform : max : 5e-2, min : 5e-6

    nOfClasses = 2, # constant
    taskType = "classification", # constant


    inputDim = 400, # constant
    hiddenDim = 400, # categ : [8, 16, 32]
    dropout = 0.0, # categ : [0, 0.1, 0.2, 0.5]

)

targetDataset = "abide1_0"
#########################


wandb.init(config = {**hyperParams, "targetDataset" : targetDataset})

#########################
# RETRIVED THE SELECTED PARAMS BACK FROM WANDB

config = wandb.config

hyperParams = Option(dict(

    batchSize = config.batchSize,
    weightDecay = config.weightDecay,
    nOfEpochs = config.nOfEpochs,

    lr = config.lr,
    minLr = config.lr * 1e-1,
    maxLr = config.lr * 2,

    nOfClasses = config.nOfClasses, 
    taskType = config.taskType,


    inputDim = config.inputDim,
    hiddenDim = int(config.hiddenDim),
    dropout = config.dropout,

))


targetDataset = config.targetDataset
#########################################


datasetIndex = int(targetDataset.split("_")[-1])
datasetName = targetDataset.split("_")[0]
datasetDetails = datasetDetailsDict[datasetName][datasetIndex]



resultss = []

for datasetSeed in datasetSeeds:
    print("running model with seed : {}".format(datasetSeed))
    results = run_brainNetCnn(hyperParams, Option({**datasetDetails,"datasetSeed": datasetSeed}), device="cuda:{}".format(device))
    resultss.append(results)

metricss = calculateMetrics(resultss) 
test_meanMetrics_seeds, test_stdMetrics_seeds, test_meanMetric_all, test_stdMetric_all = metricSummer(metricss, "test")
train_meanMetrics_seeds, train_stdMetrics_seeds, train_meanMetric_all, train_stdMetric_all = metricSummer(metricss, "train")


wandb.log({
    "test_accuracy_mean" : test_meanMetric_all["accuracy"],
    "test_roc_mean" : test_meanMetric_all["roc"],
    "test_accuracy_std" : test_stdMetric_all["accuracy"],
    "test_roc_std" : test_stdMetric_all["roc"],

    "train_accuracy_mean" : train_meanMetric_all["accuracy"],
    "train_roc_mean" : train_meanMetric_all["roc"],
    "train_accuracy_std" : train_stdMetric_all["accuracy"],
    "train_roc_std" : train_stdMetric_all["roc"]    
})










