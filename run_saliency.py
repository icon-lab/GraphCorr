
from Utils.dict2Class import Dict2Class
import torch
torch.autograd.set_detect_anomaly(True)

from Dataset.dataset import getDataset

from SaliencyNew.sal_test import test
from SaliencyNew.sal_train import train

from SaliencyMap.get_options import getOptions
from SaliencyMap.get_saliency import Saliency
from Model.saliencyModel import Model
import os


exp = "sage_hcp"
options = getOptions(exp)

torch.manual_seed(options.seed)

dataset = getDataset(options)
targetDir = options.workDir + options.logRelDir + "/" + options.expGroup + "/" + options.expName
saveName = options.saveName
# expDir = options.expGroup + "/" + options.expName
dname = options.datasets[0]

for fold in range(options.kFold):


        print("\n running fold {} on device {} \n".format(fold, options.device))
        # the trained model to be used in saliency map analysis
        modelDir = targetDir +"/fold_{}/foldModel.torch".format(fold)

        print(modelDir)

        frozenModel = torch.load(modelDir).model
        model = Model(options, frozenModel)

        testDict = Dict2Class({
                "dataset" : dataset,
                "fold" : fold,
                "model": model,
                "saveName": saveName,
                "dname": dname
        })

        train(testDict) # will do the same thing as if we are testing   
        # train and test separately   
        # test(testDict) 
        



# saliency = Saliency(options)
# distScores = saliency.getDist()
# print(distScores)


