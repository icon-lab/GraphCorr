
import pickle
import pandas
import numpy as np
import random
import torch


def loadTorchSave(atlas):

    baseFolderName = None ### replace with your data directory

    if(atlas == "AAL"):
        fileName = baseFolderName + "/hcpRest_aal.save"
    elif(atlas == "Schaefer"):
        fileName = baseFolderName + "/hcpRest_schaefer.save"


    subjectDict = torch.load(fileName)

    subjectDatas = []
    subjectIds = []

    for subjectId in subjectDict:

        subjectData = subjectDict[subjectId]
        
        if subjectData.shape[0] != 1200:
            print("Passing short subject")
            continue

        subjectIds.append(subjectId)
        subjectDatas.append(subjectData.T)


    return subjectDatas, subjectIds



def getLabels(subjectIds, targetTask):
    
    
    temp = pandas.read_csv(".../Datasets/HCP_1200/Preprocessed/pheno.csv").to_numpy() ### replace with the pheno.csv file directory
    
    phenoInfos = {}
    for row in temp:
        phenoInfos[str(row[0])] = {"gender": row[3], "age" : row[4], "fIQ" : row[121]}

    labels = []
    ages = []
    
    badSubjIds = []
    
    for subjectId in subjectIds:
        
        label = phenoInfos[subjectId][targetTask]

        agePheno = phenoInfos[subjectId]["age"]
        if("-" not in agePheno):
            age = float(agePheno.split("+")[0])
        else:    
            age = (float(agePheno.split("-")[0])) 
            
        ages.append(age)

        if(targetTask == "gender"):
            
            label = 1 if label == 'M' else 0
            
        if(targetTask == "age"):
            
            if("-" not in label):
                label = float(label.split("+")[0])
            else:    
                label = (float(label.split("-")[0]) + float(label.split("-")[1])) / 2.0
                
        if(targetTask == "fIQ"):
            if(np.isnan(label)):
                badSubjIds.append(subjectId)

        labels.append(label)

    return labels, badSubjIds, ages
    

def hcpRestLoader(atlas, targetTask):




    if(atlas == "AAL" or atlas == "Schaefer"):

        subjectDatas_, subjectIds_ = loadTorchSave(atlas)
    
    
    if(targetTask != None):    
        labels_, badSubjIds, ages_ = getLabels(subjectIds_, targetTask)

    subjectDatas = []
    subjectIds = []

    if(targetTask != None):    
        labels = []
        ages = []

        for i, subjectId in enumerate(subjectIds_):
            if(not subjectId in badSubjIds):
                subjectDatas.append(subjectDatas_[i])
                subjectIds.append(subjectIds_[i])

                ages.append(ages_[i])
                labels.append(labels_[i])

    else:

        subjectDatas = subjectDatas_
        subjectIds = subjectIds_
        
    classWeights = []
    if(targetTask == "gender"):
        for i in range(np.max(labels) + 1):
            classWeights.append(float(np.sum(np.array(labels) == i)))
        classWeights = 1/np.array(classWeights)
        classWeights = classWeights / np.sum(classWeights)                



    random.Random(12).shuffle(subjectDatas)
    random.Random(12).shuffle(subjectIds)  

    if(targetTask != None):    
        random.Random(12).shuffle(labels)
        random.Random(12).shuffle(ages)    


    
    if(targetTask != None):    
        print("hcp rest data : # subjects = {}, chance level = {}".format(len(labels), np.sum(labels) / len(labels)))

            
    if(targetTask != None):
        if(targetTask == "gender"):
            return subjectDatas, labels, subjectIds
        else:
            return subjectDatas, labels, subjectIds, classWeights, ages, None, None
    else:
        return subjectDatas, subjectIds

