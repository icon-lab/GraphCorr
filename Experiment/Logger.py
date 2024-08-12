from torch.utils.tensorboard import SummaryWriter
import telegram
from io import BytesIO
import socket
import torch
import os

from PIL import Image
import numpy as np
from matplotlib import cm
from numpy import savetxt
import matplotlib.pyplot as plt

machineName = socket.gethostname()


class Logger(object):

    def __init__(self, mainOptions):
        super().__init__()

        self.mainOptions = mainOptions

        # initing necessary logging methods

        if(mainOptions.telegramOn):
            
            token = mainOptions.botToken
            self.bot = telegram.Bot(token=token)
            self.telegramLogEpochMod = mainOptions.telegramLogEpochMod
        
        if(mainOptions.inlineOn):
            pass # nothing to do specifically for this option
        
        if(mainOptions.saveOn):
            pass # nothing to do specifically for this option

        if(mainOptions.tensorboardOn):
            self.summaryWriters = {}
            pass


        self.sendInitMessage()
        

    def log_train_epoch(self, foldNumber, epochNumber, modelInsights, metrics):
        
        dataLogs = []

        # CONSTRUCT A SINGLE LOG MESSAGE
        logMessage = "Machine : {}, ExpGroup: {}, ExpName : {}, TRAIN - fold : {}, epoch : {} ".format(machineName,self.mainOptions.expGroup, self.mainOptions.expName, foldNumber, epochNumber)
        for i, key in enumerate(metrics):
            
            logMessage += "{} : {:0.3f}".format(key, metrics[key])
            if(i != len(metrics) - 1):
                logMessage += ", "
        
        dataLogs.append({"message" : logMessage})

        # AGGREGATE INSIGHTS INTO LOG MESSAGE
        for modelInsight in modelInsights:
            if(modelInsight["type"] == "array") and (modelInsight["name"] == "lagMatrix"):

                image = self.arrayToPILImage(modelInsight["value"])
                caption = modelInsight["name"]

                dataLogs.append({"image" : image, "caption" : caption})
            
            if(modelInsight["type"] == "array") and (modelInsight["name"] == "epochNodeLog"):

                dataLogs.append({"nodeArray" : modelInsight["value"], "caption" : modelInsight["name"]})
            if(modelInsight["type"] == "array") and (modelInsight["name"] == "epochEdgeLog"):

                dataLogs.append({"edgeMatrix" : modelInsight["value"], "caption" : modelInsight["name"]})

        # NOW CALL GENERAL LOGGER FUNC
        self.log_datas("train", foldNumber, epochNumber, dataLogs)

    def log_val_epoch(self, foldNumber, epochNumber, modelInsights, metrics):

        dataLogs = []

        # CONSTRUCT A SINGLE LOG MESSAGE
        logMessage = "Machine : {}, ExpGroup: {}, ExpName : {}, VAL - fold : {}, epoch : {} ".format(machineName, self.mainOptions.expGroup, self.mainOptions.expName, foldNumber, epochNumber)
        for i, key in enumerate(metrics):
            
            logMessage += "{} : {:0.3f}".format(key, metrics[key])
            if(i != len(metrics) - 1):
                logMessage += ", "
        
        dataLogs.append({"message" : logMessage})

        # AGGREGATE INSIGHTS INTO LOG MESSAGE
        for modelInsight in modelInsights:
            if(modelInsight["type"] == "array") and (modelInsight["name"] == "lagMatrix"):

                image = self.arrayToPILImage(modelInsight["value"])
                caption = modelInsight["name"]

                dataLogs.append({"image" : image, "caption" : caption})
            
            if(modelInsight["type"] == "array") and (modelInsight["name"] == "epochNodeLog"):

                dataLogs.append({"nodeArray" : modelInsight["value"], "caption" : modelInsight["name"]})
            if(modelInsight["type"] == "array") and (modelInsight["name"] == "epochEdgeLog"):

                dataLogs.append({"edgeMatrix" : modelInsight["value"], "caption" : modelInsight["name"]})

        # NOW CALL GENERAL LOGGER FUNC
        self.log_datas("val", foldNumber, epochNumber, dataLogs)


    def log_test(self, foldNumber, modelInsights, metrics):

        dataLogs = []

        # CONSTRUCT A SINGLE LOG MESSAGE
        logMessage = "Machine : {}, ExpGroup: {}, ExpName : {}, TEST - fold : {} ".format(machineName, self.mainOptions.expGroup, self.mainOptions.expName, foldNumber)
        for i, key in enumerate(metrics):
            
            logMessage += "{} : {:0.3f}".format(key, metrics[key])
            if(i != len(metrics) - 1):
                logMessage += ", "
        
        dataLogs.append({"message" : logMessage})

        # AGGREGATE INSIGHTS INTO LOG MESSAGE
        for modelInsight in modelInsights:
            if(modelInsight["type"] == "array")and (modelInsight["name"] == "lagMatrix"):

                image = self.arrayToPILImage(modelInsight["value"])
                caption = modelInsight["name"]

                dataLogs.append({"image" : image, "caption" : caption})
            
            if(modelInsight["type"] == "array") and (modelInsight["name"] == "epochNodeLog"):

                dataLogs.append({"nodeArray" : modelInsight["value"], "caption" : modelInsight["name"]})
            if(modelInsight["type"] == "array") and (modelInsight["name"] == "epochEdgeLog"):

                dataLogs.append({"edgeMatrix" : modelInsight["value"], "caption" : modelInsight["name"]})

        # NOW CALL GENERAL LOGGER FUNC
        self.log_datas("test", foldNumber, 0, dataLogs)


    def createFoldFolders(self, foldNumber):

        trainDir = self.get_logginDir("train", foldNumber)
        valDir = self.get_logginDir("val", foldNumber)
        testDir = self.get_logginDir("test", foldNumber)

        if not os.path.exists(trainDir):
            os.makedirs(trainDir)
        if not os.path.exists(valDir):
            os.makedirs(valDir)
        if not os.path.exists(testDir):
            os.makedirs(testDir)


    def call_experimentFinished(self, allFoldMetrics):
        
        collectedMetrics = {}

        for foldMetrics in allFoldMetrics:
            for runType, metrics in foldMetrics.items():
                
                if(not runType in collectedMetrics):
                    collectedMetrics[runType] = {}
                
                for metric, value in metrics.items():
                
                    if(not metric in collectedMetrics[runType]):
                        collectedMetrics[runType][metric] = [value]
                
                    else:
                        collectedMetrics[runType][metric].append(value)


        meanMetrics =  {}
        stdMetrics = {}

        for runType, metrics in collectedMetrics.items():
            
            meanMetrics[runType] = {}
            stdMetrics[runType] = {}

            for metric, valueArray in metrics.items():
                meanMetrics[runType][metric] = np.mean(valueArray)
                stdMetrics[runType][metric] = np.std(valueArray)



        resultFile = self.get_finalResultFile()


        for runType, metrics in meanMetrics.items():

            message = "\n{} - AVERAGE \n".format(runType)
            for i, (metric, value) in enumerate(metrics.items()):
                message += "{} - {}".format(metric, value)
                if(i < len(metrics)-1):
                    message += ", "

            message += "\n\n"

            resultFile.write(message)



        for runType, metrics in stdMetrics.items():

            message = "\n{} - STD \n".format(runType)
            for i, (metric, value) in enumerate(metrics.items()):
                message += "{} - {}".format(metric, value)
                if(i < len(metrics)-1):
                    message += ", "

            message += "\n"

            resultFile.write(message)


        
        resultFile.close()

        self.sendEndMessage()

    # helpers

    def sendInitMessage(self):

        if(self.mainOptions.telegramOn):

            message = "#########START#########"
            for i in range(5):
                message += "####################### \n"
            self.bot.send_message(chat_id = self.mainOptions.chatId, text = message)    


            message = "Experiment Group : {}, Experiment : {} just started\n".format(self.mainOptions.expGroup, self.mainOptions.expName)
            self.bot.send_message(chat_id = self.mainOptions.chatId, text = message)            

    def sendEndMessage(self):

        if(self.mainOptions.telegramOn):
            message = "##########END##########"
            for i in range(5):
                message += "####################### \n"
            self.bot.send_message(chat_id = self.mainOptions.chatId, text = message)    


            message = "Experiment Group : {}, Experiment : {} just ended\n".format(self.mainOptions.expGroup, self.mainOptions.expName)
            self.bot.send_message(chat_id = self.mainOptions.chatId, text = message)            

    def log_datas(self, logType, foldNumber, epochNumber, datas):
        
        
        for data in datas:

            if("image" in data.keys()):


                image = data["image"]
                caption = data["caption"]


                if(self.mainOptions.telegramOn and epochNumber % self.mainOptions.telegramLogEpochMod == 0):

                    bio = self.imageToByteIO(image, caption)
                    self.bot.send_photo(chat_id = self.mainOptions.chatId, photo = bio, caption = caption)
                
                if(self.mainOptions.inlineOn):
                    print("\nimage : {}\n".format(caption))

                if(self.mainOptions.saveOn): 
                    
                    saveFolder = self.get_logginDir(logType, foldNumber)
                    image.save(saveFolder + "/" + caption + ".png")

            if("nodeArray" in data.keys()):

                # important nodes averaged over an epoch
                epochNodes = data["nodeArray"]
                caption = data["caption"]

                epochNodeDict = {"f":[], "m":[]}

                if(self.mainOptions.inlineOn):
                    print("\narray : {}\n".format(caption))

                if(self.mainOptions.saveOn): 
                    
                    saveFolder = self.get_logginDir(logType, foldNumber)
                    
                    epochNodes_f = epochNodes[0]
                    epochNodes_m = epochNodes[1]

                    epochNodeDict["f"].append(epochNodes_f)
                    epochNodeDict["m"].append(epochNodes_m)
                    torch.save(epochNodeDict, saveFolder + "/epochNodeDict.pt")
                    #savetxt(saveFolder + "/" + caption + ".txt", epochNodes, delimiter = ",")
                    
                    indexQueue_f = np.argsort(epochNodes_f, axis=0)[::-1] #sort from largest to smallest
                    indexQueue_m = np.argsort(epochNodes_m, axis=0)[::-1] #sort from largest to smallest

                    #savetxt(saveFolder + "/sortedNodeIndex.txt", indexQueue, delimiter=",")

                    top10indices_f = indexQueue_f[:10]
                    top10indices_f += 1 # to get the region numbers
                    lut = self.getAtlasRegions()
                    rois_f = ["important female rois"]
                    for index in top10indices_f:
                        roi = lut[int(index)]
                        rois_f.append(roi)
                    
                    top10indices_m = indexQueue_m[:10]
                    top10indices_m += 1 # to get the region numbers
                    lut = self.getAtlasRegions()
                    rois_m = ["important male rois"]
                    for index in top10indices_m:
                        roi = lut[int(index)]
                        rois_m.append(roi)
                    rois = [rois_f,rois_m]
                    savetxt(saveFolder + "/top10Regions.txt", rois, delimiter = "\n", fmt="%s")
            
            if("edgeMatrix" in data.keys()):

                epochEdges = data["edgeMatrix"]
                caption = data["caption"]

                epochEdgeDict = {"f":[], "m":[]}
                if(self.mainOptions.inlineOn):
                    print("\narray : {}\n".format(caption))

                if(self.mainOptions.saveOn): 
                    
                    saveFolder = self.get_logginDir(logType, foldNumber)
                    epochEdges_f = epochEdges[0]
                    epochEdges_m = epochEdges[1]

                    epochEdgeDict["f"].append(epochEdges_f)
                    epochEdgeDict["m"].append(epochEdges_m)
                    torch.save(epochNodeDict, saveFolder + "/epochEdgeDict.pt")

                    

                    




            elif("value" in data.keys()):
                
                if(self.mainOptions.telegramOn and epochNumber % self.mainOptions.telegramLogEpochMod == 0):

                    message = "{} : {}\n".format(data["name"], data["value"])
                    self.bot.send_message(chat_id = self.mainOptions.chatId, text = message)

                    
                if(self.mainOptions.inlineOn):
                    print("{} : {}\n".format(data["key"], data["value"]))
                
                if(self.mainOptions.saveOn):
                    pass

                if(self.mainOptions.tensorboardOn):
                    pass
            
            elif("message" in data.keys()):

                if(self.mainOptions.telegramOn and epochNumber % self.mainOptions.telegramLogEpochMod == 0):

                    self.bot.send_message(chat_id = self.mainOptions.chatId, text = data["message"])
                
                if(self.mainOptions.inlineOn):
                    
                    print("\n" + data["message"] + "\n")

                if(self.mainOptions.saveOn):
                    
                    saveLogFile = self.get_saveLogFile(logType, foldNumber)
                    saveLogFile.write("\n")
                    saveLogFile.write(data["message"])
                    saveLogFile.close()
                    print("\n\n asdf hi sdf\n\n")

                if(self.mainOptions.tensorboardOn):

                    pass
                    

    def get_logginDir(self, logType, foldNumber):
        return self.mainOptions.workDir + self.mainOptions.logRelDir + "/" + self.mainOptions.expGroup + "/" + self.mainOptions.expName + "/fold_" + str(foldNumber) + "/" + logType

    def get_saveLogFile(self, logType, foldNumber):

        assert(self.mainOptions.saveOn)
        
        logDir = self.get_logginDir(logType, foldNumber)
        textFile = open(logDir + "/" + logType + "_saveLog.txt", "a+")

        return textFile

    def get_finalResultFile(self):

        resultFilePath = self.mainOptions.workDir + self.mainOptions.logRelDir + "/" + self.mainOptions.expGroup + "/" + self.mainOptions.expName + "/results.txt"
        resultFile = open(resultFilePath, "a+")

        return resultFile

    def get_summaryWriter(self, foldNumber):

        assert(self.mainOptions.tensorboardOn)

        if(foldNumber in self.summaryWriters):
            return SummaryWriter[foldNumber]
        
        writerPath = self.mainOptions.workDir + self.mainOptions.logRelDir + "/" + self.mainOptions.expGroup + "/" + self.mainOptions.expName + "/fold_" + str(foldNumber) + "/tensorboard/"
        summaryWriter = SummaryWriter(writerPath)

        self.summaryWriters[foldNumber] = summaryWriter

        return summaryWriter

    def arrayToPILImage(self, array):

        # normalize to 0 and 1
        array += np.abs(np.min(array))
        array /= np.max(array)

        #apply colormap here
        imageArray = cm.viridis(array)
        image = Image.fromarray(np.uint8(imageArray * 255))

        return image

    def imageToByteIO(self, image, caption):
        # assumes PIL Image in the input
        bio = BytesIO()
        bio.name = caption + ".png"
        image.save(bio, "PNG")
        bio.seek(0)

        return bio
    
    def getAtlasRegions(self):
        arr = np.loadtxt("/auto/data2/irmak/ROIextracted/Atlasses/schaefer_2018/Schaefer2018_400Parcels_7Networks_order.txt", delimiter="\t", dtype=str)
        lut = {}
        for row in arr:
            roi = row[0]
            region = row[1]
            lut[int(roi)]=region
        
        return(lut)
