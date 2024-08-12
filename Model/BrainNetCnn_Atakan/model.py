from math import hypot
from Models.BrainNetCnn.code.brainNetCnn import BrainNetCNN
import torch
import numpy as np
from einops import rearrange

import time



class Model():

    def __init__(self, hyperParams, details):

        self.hyperParams = hyperParams
        self.details = details

        self.model = BrainNetCNN(hyperParams)

        # load model into gpu
        
        self.model = self.model.to(details.device)

        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss()#, weight = classWeights)

        learningRate_scaler = 1 #hyperParams.batchSize / hyperParams.nominal_batchSize
        
        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = hyperParams.lr * learningRate_scaler, weight_decay = hyperParams.weightDecay)

        # set scheduler
        steps_per_epoch = int(np.ceil(details.nOfTrains / hyperParams.batchSize))        
        
        divFactor = hyperParams.maxLr / hyperParams.lr
        finalDivFactor = hyperParams.lr / hyperParams.minLr
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, hyperParams.maxLr * learningRate_scaler, hyperParams.nOfEpochs * steps_per_epoch, div_factor=divFactor, final_div_factor=finalDivFactor)
        
        
    def step(self, x, y, train=True):

        """
            x = (batchSize, N, N) # connectivity matrixes 
            y = (batchSize, numberOfClasses)

        """

        # PREPARE INPUTS
        
        inputs, y = self.prepareInput(x, y)

        # DEFAULT TRAIN ROUTINE
        
        if(train):
            self.model.train()
        else:
            self.model.eval()

        yHat = self.model(*inputs)
        loss = self.getLoss(yHat, y)

        preds = yHat.argmax(1)
        probs = yHat.softmax(1)

        if(train):

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()   

            if(not isinstance(self.scheduler, type(None))):
                self.scheduler.step()               

        loss = loss.detach().to("cpu")
        preds = preds.detach().to("cpu")
        probs = probs.detach().to("cpu")

        y = y.to("cpu")
        
        torch.cuda.empty_cache()


        return loss, preds, probs, y
        


    # HELPER FUNCTIONS HERE

    def prepareInput(self, x, y):

        """
            x = (batchSize, N, N) # connectivity matrixes
            y = (batchSize, )

        """
        # to gpu now

        x = x.to(self.details.device)
        y = y.to(self.details.device)


        return (x, ), y

    def getLoss(self, yHat, y):
        
        cross_entropy_loss = self.criterion(yHat, y)

        return cross_entropy_loss 


