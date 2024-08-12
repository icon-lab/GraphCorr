
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from random import shuffle, randrange
import numpy as np
import random

from .Loaders.hcpRestNew import hcpRestLoader


loaderMapper = {
    "hcpRest" : hcpRestLoader,
    # add other datasets if you want
}

def getDataset(mainOptions):

    # if(mainOptions.supervision == "supervised"):
    return SupervisedDataset(mainOptions)



class SupervisedDataset(Dataset):
    
    def __init__(self, mainOptions):

        self.batchSize = mainOptions.batchSize
        self.dynamicLength = mainOptions.dynamicLength
        self.foldCount = mainOptions.kFold

        loader = loaderMapper[mainOptions.datasets[0]]

        self.kFold = StratifiedKFold(mainOptions.kFold, shuffle=True, random_state=0) if mainOptions.kFold is not None else None
        self.k = None

        self.data, self.labels, self.subjectIds = loader(mainOptions.atlas, mainOptions.targetTask)
        
        self.targetData = None
        self.targetLabel = None

    def __len__(self):
        return len(self.data) if isinstance(self.targetData, type(None)) else len(self.targetData)

    def get_nOfTrains_perFold(self):
        
        return len(self.data)        

    def setFold(self, fold, train=True):

        self.k = fold
        self.train = train

        if(self.kFold == None): # if this is the case, train must be True
            trainIdx = list(range(len(self.data)))
        else:
            trainIdx, testIdx = list(self.kFold.split(self.data, self.labels))[fold]        

        random.Random(12).shuffle(trainIdx)

        self.targetData = [self.data[idx] for idx in trainIdx] if train else [self.data[idx] for idx in testIdx]
        self.targetLabels = [self.labels[idx] for idx in trainIdx] if train else [self.labels[idx] for idx in testIdx]
        self.targetSubjIds = [self.subjectIds[idx] for idx in trainIdx] if train else [self.subjectIds[idx] for idx in testIdx]

    def getFold(self, fold, train=True):
        
        self.setFold(fold, train)

        if(train):
            return DataLoader(self, batch_size=self.batchSize, shuffle=False)
        else:
            return DataLoader(self, batch_size=1, shuffle=False)            


    def __getitem__(self, idx):

        subject = self.targetData[idx]
        label = self.targetLabels[idx]
        subjId = self.targetSubjIds[idx]

        # normalize timeseries
        timeseries = subject # (numberOfRois, time)
        timeseries = (timeseries - np.mean(timeseries, axis=1, keepdims=True)) / np.std(timeseries, axis=1, keepdims=True)

        # dynamic sampling if train
        if(self.train):
            
            if(timeseries.shape[1] < self.dynamicLength):
                print(timeseries.shape[1], self.dynamicLength)

            samplingInit = 0 if timeseries.shape[1] == self.dynamicLength else randrange(timeseries.shape[1] - self.dynamicLength)
            timeseries = timeseries[:, samplingInit : samplingInit + self.dynamicLength]

        return {"timeseries" : timeseries.astype(np.float32), "label" : label, "subjId" : subjId}







