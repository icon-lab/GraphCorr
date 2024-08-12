import os

from scipy import ma
from sklearn import metrics
from glob import glob
import torch

from Experiment.Logger import Logger

class ModelTracker(object):
    def __init__(self, mainOptions, trackerOptions):

        self.mainOptions = mainOptions
        self.trackerOptions = trackerOptions

        self.model = trackerOptions.model

        self.logger = Logger(mainOptions)

        self.optimizer = None
        self.scheduler = None

        self.lastEpochMetrics = {
            "train" : None,
            "validation" : None,
            "test" : None
        }

        self.stateTracker = {}

        # temporary variables
        self.currentFold = None
        self.currentEpoch = -1        
        self.currentModel = None



    # LOG HANDLERS


    def log_epoch_train(self, model_train_insights, loss, preds=None, probs=None, groundTruth=None):
        
        # calculate metrics

        if(self.mainOptions.taskType == "classification"):

            accuracy, precision, recall, roc_auc = self.calculate_basicStats(preds, probs, groundTruth)        

            metrics = {"accuracy" : accuracy, "precision" : precision, "recall" : recall, "roc_auc" : roc_auc, "loss" : loss}

        
        elif(self.mainOptions.taskType == "regression"):

            metrics = {"loss" : loss}

        self.insertIntoStateTracker("train", metrics)

        self.logger.log_train_epoch(self.currentFold, self.currentEpoch, model_train_insights, metrics)



    def log_epoch_val(self, model_val_insight, loss, preds=None, probs=None, groundTruth=None):

        if(self.mainOptions.taskType == "classification"):
            
            accuracy, precision, recall, roc_auc = self.calculate_basicStats(preds, probs, groundTruth)

            metrics = {"accuracy" : accuracy, "precision" : precision, "recall" : recall, "roc_auc" : roc_auc, "loss" : loss}        

        elif(self.mainOptions.taskType == "regression"):

            metrics = {"loss" : loss}
            
        self.insertIntoStateTracker("validation", metrics)

        self.logger.log_val_epoch(self.currentFold, self.currentEpoch, model_val_insight, metrics)
        
        
    
    def log_test(self, model_test_insight, loss, preds=None, probs=None, groundTruth=None):

        if(self.mainOptions.taskType == "classification"):            

            accuracy, precision, recall, roc_auc = self.calculate_basicStats(preds, probs, groundTruth)

            metrics = {"accuracy" : accuracy, "precision" : precision, "recall" : recall, "roc_auc" : roc_auc, "loss" : loss}        

        elif(self.mainOptions.taskType == "regression"):

            metrics = {"loss" : loss}

        self.insertIntoStateTracker("test", metrics)

        self.logger.log_test(self.currentFold, model_test_insight, metrics)


    def finalizeTrainFold(self):

        self.delete_prevEpochModel()
        self.saveFoldModel()

        self.currentFold = None
        self.currentEpoch = -1
        self.currentModel = None        


    def finalizeExperiment(self):

        finalFoldMetrics = []

        kFold = self.mainOptions.kFold if self.mainOptions.kFold != None else 1

        for fold in range(kFold):
            finalFoldMetrics.append(self.collectLastMetrics_fromStateTracker(fold))
            

        self.logger.call_experimentFinished(finalFoldMetrics)

    

    # NEW POINT HANDLERS



    def train_setEpoch(self, epoch):

        if(self.currentModel != None):
            self.saveEpochModel()

        self.currentEpoch = epoch
        
    
    def train_setFold(self, fold, nOfTrains):
        
        self.currentFold = fold      

        model = self.getCurrentEpochModel()

        if(model == None):
            self.trackerOptions.nOfTrains = nOfTrains
            model = self.model(self.mainOptions, self.trackerOptions, self.currentFold)
        
        self.currentModel = model

        # finally notify logger
        self.logger.createFoldFolders(fold)

        return model

    
    def test_setFold(self, fold):
        
        self.currentFold = fold   
        model = self.loadFoldModel()

        return model     


    # HELPER FUNCTIONS

    def collectLastMetrics_fromStateTracker(self, fold):

        willReturn = {"train" : {} if len(self.stateTracker[fold]["train"]) == 0 else self.stateTracker[fold]["train"][-1],\
                    "validation" : {} if len(self.stateTracker[fold]["validation"]) == 0 else self.stateTracker[fold]["validation"][-1],\
                    "test" : {} if len(self.stateTracker[fold]["test"]) == 0 else self.stateTracker[fold]["test"][-1]}

        return willReturn

    def insertIntoStateTracker(self,runType, metrics):

        if(not self.currentFold in self.stateTracker):
            self.stateTracker[self.currentFold] = {"train" : [], "validation" : [], "test" : []}
        
        self.stateTracker[self.currentFold][runType].append(metrics)


    def get_fullAndSemi_finisheds(self):

        anyFoldFolders = glob(self.mainOptions.workDir + self.mainOptions.logRelDir + "/" + self.mainOptions.expGroup + "/" + self.mainOptions.expName + "/fold*")

        if(len(anyFoldFolders) == 0):
            return None

        if(self.mainOptions.kFold != None):

            doneFolds = []
            semiFinishedFold = (None, None) # foldNuber, lastFinished_epochNumber

            for foldFolder in anyFoldFolders:
                if(os.path.exists(foldFolder + "/foldModel.torch")):    
                    doneFolds.append(self.getFoldNumber_fromFolderName(foldFolder))
                else:

                    savedEpochModel = glob(foldFolder + "/*epochModel.torch")

                    if(len(savedEpochModel) != 0):

                        lastFinished_epochNumber = int(savedEpochModel[0].split("/")[-1].split("_")[0])
                        semiFinishedFold = (self.getFoldNumber_fromFolderName(foldFolder), lastFinished_epochNumber)
                    
            return doneFolds, semiFinishedFold

        else:

            foldFolder = anyFoldFolders[0]

            if(os.path.exists(foldFolder + "/foldModel.torch")):
                print("Training already done mate, quitting")
                lastFinished_epochNumber = None
            
            else:
                savedEpochModel = glob(foldFolder + "/*epochModel.torch")

                if(len(savedEpochModel) != 0):

                    lastFinished_epochNumber = int(savedEpochModel[0].split("/")[-1].split("_")[0])
                else: 
                    lastFinished_epochNumber = -1

            return lastFinished_epochNumber

    def saveEpochModel(self):

        self.delete_prevEpochModel()

        saveFolder = self.getCurrentFoldFolderName()
        os.makedirs(saveFolder, exist_ok=True)

        with open(saveFolder + "/" + str(self.currentEpoch) + "_epochModel.torch", "wb") as handle:
            print(self.currentEpoch, self.currentModel)
            torch.save(self.currentModel, handle)



    def loadEpochModel(self):
        
        saveFolder = self.getCurrentFoldFolderName()

        with open(saveFolder + "/" + str(self.currentEpoch) + "_epochModel.torch", "rb") as handle:
            model = torch.load(handle)

        return model



    def saveFoldModel(self):

        saveFolder = self.getCurrentFoldFolderName()
        os.makedirs(saveFolder, exist_ok=True)

        with open(saveFolder + "/foldModel.torch", "wb") as handle:
            torch.save(self.currentModel, handle)



    def loadFoldModel(self):

        saveFolder = self.getCurrentFoldFolderName()

        with open(saveFolder + "/foldModel.torch", "rb") as handle:
            model = torch.load(handle)

        return model



    def getCurrentFoldFolderName(self):

        return self.mainOptions.workDir + self.mainOptions.logRelDir + "/" + self.mainOptions.expGroup + "/" + self.mainOptions.expName + "/fold_" + str(self.currentFold)


    def getFoldNumber_fromFolderName(self, folderName):

        return int(folderName.split("/")[-1].split("_")[-1])

    def delete_prevEpochModel(self):
        
        currentFolder = self.getCurrentFoldFolderName()
        epochModelFile = currentFolder + "/" + str(self.currentEpoch - 1) + "_epochModel.torch"

        if(os.path.exists(epochModelFile)):
            os.remove(epochModelFile)



    def getCurrentEpochModel(self):

        saveFolder = self.getCurrentFoldFolderName()

        epochModelPath = glob(saveFolder + "/*_epochModel.torch")
        
        if(len(epochModelPath) > 0):
            with open(epochModelPath[0], "rb") as handle:
                return torch.load(handle)

        return None


    def calculate_basicStats(self, preds, probs, groundTruth):
        
        accuracy = metrics.accuracy_score(groundTruth, preds)
        precision = metrics.precision_score(groundTruth, preds, average='binary' if self.mainOptions.nOfClasses==2 else 'micro')
        recall = metrics.recall_score(groundTruth, preds, average='binary' if self.mainOptions.nOfClasses==2 else 'micro')
        roc_auc = metrics.roc_auc_score(groundTruth, probs[:,1]) if self.mainOptions.nOfClasses  == 2 else \
                metrics.roc_auc_score(groundTruth, probs, average="macro", multi_class="ovr")

        return accuracy, precision, recall, roc_auc
