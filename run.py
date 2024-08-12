


from Utils.dict2Class import Dict2Class
import torch
torch.autograd.set_detect_anomaly(True)

from Dataset.dataset import getDataset
from Experiment.ModelTracker import ModelTracker

from Experiment.train import train
from Experiment.test import test

import os
import shutil


def prepareExperimentDir(options, cleanPreviousDir):
        
    # create experiment directory
    targetDir = options.workDir + options.logRelDir + "/" + options.expGroup + "/" + options.expName

    print(targetDir)
    
    if(cleanPreviousDir and os.path.exists(targetDir)):
        shutil.rmtree(targetDir)
        print("deleting {}".format(targetDir))

    os.makedirs(targetDir, exist_ok=True)

    # write options
    optionLogFile = open(targetDir + "/options.txt", "w")
    
    for item in list(vars(options).items()):
        optionLogFile.write("{} : {}\n".format(item[0], item[1]))
    
    optionLogFile.close()
    
    # freeze existing code

    if(os.path.exists(targetDir + "/frozenCode")):
        shutil.rmtree(targetDir + "/frozenCode")
        
    os.makedirs(targetDir + "/frozenCode", exist_ok=True)
    
    shutil.copytree(options.workDir + "Dataset", targetDir+"/frozenCode/Dataset")
    shutil.copytree(options.workDir + "Model", targetDir+"/frozenCode/Model")
    shutil.copytree(options.workDir + "Experiment", targetDir+"/frozenCode/Experiment")    
    shutil.copytree(options.workDir + "Utils", targetDir+"/frozenCode/Utils")    
    
    shutil.copyfile(options.workDir + "/run.py", targetDir + "/frozenCode/run.py")
    


def experimentRunner(options, cleanPreviousDir=False):

    prepareExperimentDir(options, cleanPreviousDir)


    from Model.model import Model


    # SET DATASET

    dataset = getDataset(options)

    # SET TRACKER

    trackerOptions = Dict2Class({
            "model" : Model,
            "nOfTrains" : dataset.get_nOfTrains_perFold()
    })

    modelTracker = ModelTracker(options, trackerOptions)

    # RUN TRAIN


    targetFolds = range(options.kFold)

    previousRuns = modelTracker.get_fullAndSemi_finisheds()

    if(previousRuns != None):
            
            (doneFolds, (semiFinishedFold, lastDoneEpoch)) = previousRuns 

            # first handle non finished fold, if there is any
            if(semiFinishedFold != None):
                    trainDict = Dict2Class({
                            "dataset" :  dataset,
                            "targetFolds" : [semiFinishedFold],
                            "startEpoch" : lastDoneEpoch+1,
                            "totalEpoch" : options.nOfEpochs,
                            "modelTracker" : modelTracker,
                            "device" : options.device,
                    })

                    testDict = Dict2Class({
                            "dataset" : dataset,
                            "targetFolds" : [semiFinishedFold],
                            "modelTracker" : modelTracker,
                            "device" : options.device,
                    })        

                    train(trainDict)
                    test(testDict)
                    
            # now it is finished
            doneFolds.append(semiFinishedFold)

    else:    
            doneFolds = []

    # handle rest of the folds
    targetFolds = [k for k in targetFolds if k not in doneFolds]

    for fold in targetFolds:

            trainDict = Dict2Class({
                    "dataset" : dataset,
                    "targetFolds" : [fold],
                    "startEpoch" : 0,
                    "totalEpoch" : options.nOfEpochs,
                    "modelTracker" : modelTracker,
                    "device" : options.device,
            })

            testDict = Dict2Class({
                    "dataset" : dataset,
                    "targetFolds" : [fold],
                    "modelTracker" : modelTracker,
                    "device" : options.device,
            })

            train(trainDict)
            test(testDict)



    modelTracker.finalizeExperiment()