
from math import hypot
from Models.BrainGnn.code.brainGnn import BrainGnn
import torch
import numpy as np
from einops import rearrange

import torch.nn.functional as F 

import time

EPS = 1e-10

from .brainGnn import BrainGnn
from .graphConstructor import calFcMatrixes, getGraphicalData_fromFcMatrixes

class Model():

    def __init__(self, hyperParams, details):

        self.hyperParams = hyperParams
        self.details = details

        self.model = BrainGnn(hyperParams)

        # load model into gpu
        self.device = details.device
        self.model = self.model.to(details.device)

        # set criterion
        classWeights = torch.tensor(details.classWeights).float().to(details.device)
        self.criterion = torch.nn.CrossEntropyLoss()#, weight = classWeights)
      
        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = hyperParams.lr, weight_decay = hyperParams.weightDecay)

        learningRate_scaler = 1 #hyperParams.batchSize / hyperParams.nominal_batchSize
        
        # set scheduler
        steps_per_epoch = int(np.ceil(details.nOfTrains / hyperParams.batchSize))        
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, mainOptions.nOfEpochs * steps_per_epoch, mainOptions.minLr * learningRate_scaler)
        
        divFactor = hyperParams.maxLr / hyperParams.lr
        finalDivFactor = hyperParams.lr / hyperParams.minLr
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, hyperParams.maxLr * learningRate_scaler, hyperParams.nOfEpochs * (steps_per_epoch), div_factor=divFactor, final_div_factor=finalDivFactor)
        

    def step(self, x, y, train=True):

        """
            x = (batchSize, N, dynamicLength) 
            y = (batchSize, numberOfClasses)

        """

        # PREPARE INPUTS
        
        inputs, y = self.prepareInput(x, y)

        # DEFAULT TRAIN ROUTINE
        
        if(train):
            self.model.train()
        else:
            self.model.eval()

        output, w1, w2, s1, s2 = self.model(*inputs)
        loss = self.getLoss(output, w1, w2, s1, s2, y)

        preds = output.argmax(1)
        probs = torch.exp(output)

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
        
        Input:
            x = (batchSize, N, T)
            y = (batchSize, )
        
        Output:
            (
                out1 = nodeFeatures_graph = (batchSize * N, N)
                out2 = edge_index_graph = (2, batchSize * N #ofConnectionsPerRoi
                out3 = batch = (batchSize * N)
            )

            y = (batchSize, )

        """
        # to gpu now

        batchSize = x.shape[0]
        N = x.shape[1]

        Fc_sparse = calFcMatrixes(x.permute((0,2,1)).numpy(), "partial correlation") # (batchSize, N, N)
        Fc_corr = calFcMatrixes(x.permute((0,2,1)).numpy(), "correlation") # (batchSize, N, N)

        graphicalData = getGraphicalData_fromFcMatrixes(fcMatrixes=Fc_sparse, nodeFeatures=Fc_corr, sparsity=self.hyperParams.sparsity, edgeMethod=self.hyperParams.edgeMethod)
        
        x_graph = graphicalData.x.to(self.details.device)
        edge_index = graphicalData.edge_index.to(self.details.device)
        edge_attr = graphicalData.edge_attr.to(self.details.device)
        batch = graphicalData.batch.to(self.details.device)
        pos = graphicalData.pos.to(self.details.device)

        y = y.to(self.details.device)
        
        return (x_graph, edge_index, edge_attr, batch, pos), y


    def getLoss(self, output, w1, w2, s1, s2, y):
        
        loss_c = F.nll_loss(output, y)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = self.topk_loss(s1,self.hyperParams.poolingRatio)
        loss_tpk2 = self.topk_loss(s2,self.hyperParams.poolingRatio)
        loss_consist = 0
        for c in range(self.hyperParams.nOfClasses):
            loss_consist += self.consist_loss(s1[y == c])
        loss = self.hyperParams.lamb0*loss_c + self.hyperParams.lamb1 * loss_p1 + self.hyperParams.lamb2 * loss_p2 \
                    + self.hyperParams.lamb3 * loss_tpk1 + self.hyperParams.lamb4 *loss_tpk2 + self.hyperParams.lamb5* loss_consist

        return loss

    
    def topk_loss(self, s,ratio):
        if ratio > 0.5:
            ratio = 1-ratio
        s = s.sort(dim=1).values
        res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
        return res


    def consist_loss(self, s):
        if len(s) == 0:
            return 0
        s = torch.sigmoid(s)
        W = torch.ones(s.shape[0],s.shape[0])
        D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
        L = D-W
        L = L.to(self.device)
        res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
        return res
