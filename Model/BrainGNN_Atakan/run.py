from tqdm import tqdm
import torch
import numpy as np
import random
import os 
import sys
import time

if(not "utils" in os.getcwd()):
    sys.path.append("../../../")

from utils import Option, calculateMetric
from Models.BrainGnn.code.model import Model
from Dataset.dataset import getDataset


def train(model, dataset, taskType, fold, nOfEpochs):

    dataLoader = dataset.getFold(fold, train=True)

    if(taskType == "classification"):

        
        for epoch in range(nOfEpochs):

                preds = []
                probs = []
                groundTruths = []
                losses = []

                for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
                    
                    time1 = time.time()
                    xTrain = data["timeseries"]
                    yTrain = data["label"] # (batchSize, )
                    time2 = time.time()
                    # NOTE: xTrain and yTrain are still on "cpu" at this point

                    train_loss, train_preds, train_probs, yTrain = model.step(xTrain, yTrain, train=True)
                    time3 = time.time()

                    #print("time2-time1 = {}, time3-time2 = {}".format(time2-time1, time3-time2))

                    torch.cuda.empty_cache()

                    preds.append(train_preds)
                    probs.append(train_probs)
                    groundTruths.append(yTrain)
                    losses.append(train_loss)

                preds = torch.cat(preds, dim=0).numpy()
                probs = torch.cat(probs, dim=0).numpy()
                groundTruths = torch.cat(groundTruths, dim=0).numpy()
                losses = torch.tensor(losses).numpy()

                metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
                print("Train metrics : {}".format(metrics))


        return preds, probs, groundTruths, losses

    elif(taskType == "regression"):

        for epoch in range(nOfEpochs):

            losses = []

            for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
                
                xTrain = data["timeseries"]

                # NOTE: xTrain is still on "cpu" at this point

                train_loss = model.step(xTrain, train=True)

                torch.cuda.empty_cache()

                losses.append(train_loss)


            losses = torch.tensor(losses).numpy()

        return losses,


def test(model, dataset, taskType, fold):

    dataLoader = dataset.getFold(fold, train=False)

    if(taskType == "classification"):

        preds = []
        probs = []
        groundTruths = []
        losses = []        

        for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'Testing fold:{fold}')):

            xTest = data["timeseries"]
            yTest = data["label"]

            # NOTE: xTrain and yTrain are still on "cpu" at this point

            test_loss, test_preds, test_probs, yTest = model.step(xTest, yTest, train=False)
            
            torch.cuda.empty_cache()

            preds.append(test_preds)
            probs.append(test_probs)
            groundTruths.append(yTest)
            losses.append(test_loss)

        preds = torch.cat(preds, dim=0).numpy()
        probs = torch.cat(probs, dim=0).numpy()
        groundTruths = torch.cat(groundTruths, dim=0).numpy()
        loss = torch.tensor(losses).numpy().mean() 

        metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
        print("Test metrics : {}".format(metrics))                 
        
        return preds, probs, groundTruths, loss
        
    elif(taskType == "regression"):

        losses = []        

        for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'Testing fold:{fold}')):

            xTest = data["timeseries"]

            # NOTE: xTrain is still on "cpu" at this point

            test_loss = model.step(xTest, train=False)
            
            torch.cuda.empty_cache()

            losses.append(test_loss)

        losses = torch.tensor(losses).numpy()            
            
        return losses




def run_brainGnn(hyperParams, datasetDetails, device="cuda:0", save=False):


    # extract datasetDetails

    targetDataset = datasetDetails.datasetName
    targetTask = datasetDetails.targetTask
    atlas = datasetDetails.atlas
    foldCount = datasetDetails.foldCount
    datasetSeed = datasetDetails.datasetSeed
    dynamicLength = datasetDetails.dynamicLength


    batchSize = hyperParams.batchSize
    taskType = hyperParams.taskType

    dataset = getDataset(Option({

        "batchSize" : batchSize,
        "dynamicLength" : dynamicLength,
        "foldCount" : foldCount,
        "datasetSeed" : datasetSeed,
        
        "targetTask" : targetTask,
        "atlas" : atlas,
        "targetDataset" : targetDataset

    }))

    nOfEpochs = hyperParams.nOfEpochs

    details = Option({
        "device" : device,
        "classWeights" : dataset.getClassWeights(),
        "nOfTrains" : dataset.get_nOfTrains_perFold(),
    })


    results = []


    for fold in range(foldCount):

        model = Model(hyperParams, details)

        if(taskType == "classification"):

            train_preds, train_probs, train_groundTruths, train_loss = train(model, dataset, taskType, fold, nOfEpochs)   
            test_preds, test_probs, test_groundTruths, test_loss = test(model, dataset, taskType, fold)

            result = {

                "train" : {
                    "labels" : train_groundTruths,
                    "predictions" : train_preds,
                    "probs" : train_probs,
                    "loss" : train_loss
                },

                "test" : {
                    "labels" : test_groundTruths,
                    "predictions" : test_preds,
                    "probs" : test_probs,
                    "loss" : test_loss
                }

            }

            results.append(result)

        elif(taskType == "regression"):
            raise("Regression not implemented yet") 


    return results
