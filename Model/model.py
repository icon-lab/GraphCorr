
from numpy import isin
from .graph import GraphNN
import torch
from torch import nn
import numpy as np
from einops import repeat, rearrange

from .windower import calWindowedFcMatrixes, getGraphicalData_fromFcMatrixes, calFcMatrixes, windowBoldSignal
from .DistanceMatrix.getDistanceEdges import getDistanceEdges, getDistanceMatrix
import torch.nn.functional as F 

import time

EPS = 1e-10

class Model():

    def __init__(self, mainOptions, details, currentFold):

        self.mainOptions = mainOptions

        self.model = GraphNN(mainOptions)

        self.model = self.model.to(mainOptions.device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


        learningRate_scaler = mainOptions.batchSize / mainOptions.nominal_batchSize

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = mainOptions.lr * learningRate_scaler, weight_decay = mainOptions.weightDecay)

        divFactor = self.mainOptions.maxLr / self.mainOptions.lr
        finalDivFactor = self.mainOptions.lr / self.mainOptions.minLr
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, mainOptions.maxLr * learningRate_scaler, mainOptions.nOfEpochs * details.nOfTrains, div_factor=divFactor, final_div_factor=finalDivFactor)
        
    def step(self, x, y, train=True):

        """
            x = (batchSize, N, dynamicLength) 
            y = (batchSize, numberOfClasses)
        
        """

        
        inputs, y = self.prepareInput(x, y)

        # DEFAULT TRAIN ROUTINE

        if(train):
            self.model.train()
        else:
            self.model.eval()

        if (self.mainOptions.graphLayer == "brainGnn"):

            (yHat, w1, w2, s1, s2), lagMatrix = self.model(*inputs)
            
            loss = self.getLoss_brainGnn(yHat, w1, w2, s1, s2, y)
        else:

            yHat, lagMatrix = self.model(*inputs)

            loss = self.getLoss(yHat, y)

        preds = yHat.argmax(1)
        probs = yHat.softmax(1)


        if(train):

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.scheduler.step()
        
        if(not isinstance(lagMatrix, type(None))):
            modelInsights = self.getModelInsights(y,lagMatrix=lagMatrix)
        else:
            modelInsights = []
                    
        loss = loss.detach().to("cpu")
        preds = preds.detach().to("cpu")
        probs = probs.detach().to("cpu")

        y = y.to("cpu")

        torch.cuda.empty_cache()
        
        
        return loss, preds, probs, modelInsights, y            

    

    def prepareInput(self, x, y):
        """
        INPUT
        
            x = (batchSize, N, T)
            y = (batchSize, )
        
        OUTPUT

            (

                out1 = nodeFeatures_graph = (batchSize * #windows * N, N) 
                out2 = edge_index_graph = (2, batchSize * #windows * N * #ofConnectionsPerRoi_graph)
                out3 = batch = (batchSize * #windows * N)
                
                out4 = nodeFeatures_corr = (batchSize * #windows * N, Tw) # Tw = length of a window
                out5 = edge_index_corr = (2, batchSize * #windows * N * #ofConnectionsPerRoi_corr)
                out6 = prior_connectivityMatrix = (batchSize * #windows, N, N)
            ),

            y = (batchSize, )

        """

        batchSize = x.shape[0]
        N = x.shape[1]
        T = x.shape[2]
        # for BrainGNN
        
        Fc_corr = calFcMatrixes(x.permute((0,2,1)).numpy(), "correlation") # (batchSize, N, N)
        Fc_sparse = Fc_corr #calFcMatrixes(x.permute((0,2,1)).numpy(), "partial correlation") # (batchSize, N, N)
        Fc_sparse = torch.from_numpy(Fc_sparse)
        Fc_corr = torch.from_numpy(Fc_corr)

        windowedX, _ = windowBoldSignal(x, T, self.mainOptions.windowSize, self.mainOptions.stride)
        # windowedX shape = (batchSize, #windows, N, windowLength)
        windowedFC = calWindowedFcMatrixes(windowedX) # of shape = (batchSize, #windows, N, N)
        # print(windowedFC.shape[1])
        extX = x.unsqueeze(dim = 1) # batchSize, 1, N, T
        fcMatrixes = calWindowedFcMatrixes(extX) # of shape = (batchSize, 1, N, N)

        statDynFC = torch.cat((windowedFC,fcMatrixes),dim=1)
        statDynFC = rearrange(statDynFC, "b w n m -> (b w) n m").unsqueeze(dim=1)
        windowCount = fcMatrixes.shape[1] # 1 window at start ( 1 x batch graphs will be generated)

        graphicalData_bgnn = getGraphicalData_fromFcMatrixes(Fc_sparse.unsqueeze(dim=1), Fc_corr.unsqueeze(dim=1), self.mainOptions.brainGnn.sparsity, "connection_thresholded")
        
        graphicalData_corr = getGraphicalData_fromFcMatrixes(fcMatrixes, extX, self.mainOptions.sparsity_corr, "connection_thresholded")

        if(self.mainOptions.useStatDynFC):
            graphicalData_graph = getGraphicalData_fromFcMatrixes(statDynFC, statDynFC, self.mainOptions.sparsity_graph, "value_thresholded")
        else:
            graphicalData_graph = getGraphicalData_fromFcMatrixes(fcMatrixes, fcMatrixes, self.mainOptions.sparsity_graph, "value_thresholded")

        

        # load these to gpu now
        if(self.mainOptions.useGraphCorr):


            # load necessaries to gpu
            x_corr = graphicalData_corr.x.to(self.mainOptions.device)
            edge_index_corr = graphicalData_corr.edge_index.to(self.mainOptions.device)
            prior_connectivityMatrix = rearrange(windowedFC, "b w n m -> (b w) n m").to(self.mainOptions.device)
            
            if (self.mainOptions.graphLayer == "brainGnn"):
                edge_index_graph = graphicalData_bgnn.edge_index.to(self.mainOptions.device)
            else:
                edge_index_graph = graphicalData_graph.edge_index.to(self.mainOptions.device)
            
            batch = graphicalData_graph.batch.to(self.mainOptions.device) # does not matter where it comes from, graphicalData_corr is also same
            
            y = y.to(self.mainOptions.device)
            
            # needed for BrainGNN
            edge_attr = graphicalData_bgnn.edge_attr.to(self.mainOptions.device)
            pos = graphicalData_bgnn.pos.to(self.mainOptions.device)

            return (
                None, # x_graph
                edge_index_graph,
                batch,

                edge_attr,
                pos,

                x_corr,
                edge_index_corr,
                prior_connectivityMatrix

            ), y 

        elif(self.mainOptions.graphLayer == "brainNetCnn"):

            x = fcMatrixes.to(self.mainOptions.device)
            y = y.to(self.mainOptions.device)
            return (
                x,
                None,
                None,
                
                None,
                None,
                
                None,
                None,
                None
             ), y

        else:

            if (self.mainOptions.graphLayer == "brainGnn"):
                x_graph = graphicalData_bgnn.x.to(self.mainOptions.device)
                edge_index_graph = graphicalData_bgnn.edge_index.to(self.mainOptions.device)
                batch = graphicalData_bgnn.batch.to(self.mainOptions.device) 
            else:
                x_graph = graphicalData_graph.x.to(self.mainOptions.device)
                
                edge_index_graph = graphicalData_graph.edge_index.to(self.mainOptions.device)
                batch = graphicalData_graph.batch.to(self.mainOptions.device) # does not matter where it comes from, graphicalData_corr is also same
            
            y = y.to(self.mainOptions.device)
            
            # needed for BrainGNN
            edge_attr = graphicalData_bgnn.edge_attr.to(self.mainOptions.device)
            pos = graphicalData_bgnn.pos.to(self.mainOptions.device)

            return (
                x_graph,
                edge_index_graph,
                batch,

                edge_attr,
                pos,

                None, # x_corr
                None, # edge_index_corr
                None # prior_connectivityMatrix

            ), y


    def getLoss(self, yHat, y):
        
        cross_entropy_loss = self.criterion(yHat, y)

        return cross_entropy_loss

    def getLoss_brainGnn(self, output, w1, w2, s1, s2, y):
        
        loss_c = F.nll_loss(output, y)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = self.topk_loss(s1,self.mainOptions.brainGnn.poolingRatio)
        loss_tpk2 = self.topk_loss(s2,self.mainOptions.brainGnn.poolingRatio)
        loss_consist = 0
        for c in range(self.mainOptions.nOfClasses):
            loss_consist += self.consist_loss(s1[y == c])
        loss = self.mainOptions.brainGnn.lamb0*loss_c + self.mainOptions.brainGnn.lamb1 * loss_p1 + self.mainOptions.brainGnn.lamb2 * loss_p2 \
                    + self.mainOptions.brainGnn.lamb3 * loss_tpk1 + self.mainOptions.brainGnn.lamb4 *loss_tpk2 + self.mainOptions.brainGnn.lamb5* loss_consist

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
        L = L.to(self.mainOptions.device)
        res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
        return res

    def getModelInsights(self, y, **kwargs):
        
        insights = []
        
        for key in kwargs.keys():

            value = kwargs[key]

            if(key == "lagMatrix"):

                if(value.shape[0] * value.shape[1] == 0):
                    continue

                value = np.repeat(value, 30, axis=0)
                value = np.repeat(value, 30, axis=1)            
            
            if(key == "subgraph_nodes"):
                
                nodes, edges = value
                
            
                # value has (topk pooling num nodes * batch size) entries
                allNodes_f = np.zeros(self.mainOptions.nOfRois)
                allNodes_m = np.zeros(self.mainOptions.nOfRois)

                for val in nodes:

                    graphNum = val // self.mainOptions.nOfRois
                    nodeNum = val % self.mainOptions.nOfRois

                    if y[graphNum] == 0:
                        allNodes_f[nodeNum] += 1
                    else:
                        allNodes_m[nodeNum] += 1
                # normalize
                numMales = torch.sum(y).item()
                numFemales = self.mainOptions.batchSize- numMales

                numMales = 1 if numMales == 0 else numMales
                numFemales = 1 if numFemales == 0 else numFemales

                allNodes_f /= numFemales
                allNodes_m /= numMales
                value = [allNodes_f, allNodes_m]

                
                allEdges_f = np.zeros((self.mainOptions.nOfRois,self.mainOptions.nOfRois))
                allEdges_m = np.zeros((self.mainOptions.nOfRois,self.mainOptions.nOfRois))
                

                for i in range(edges.shape[1]):
                    connection = edges[:,i]
                    subj = connection[0] // 5 # top k nodes
                    connectedNodes = nodes[connection]

                    connectedNodes = connectedNodes % self.mainOptions.nOfRois

                    if y[subj] == 0:
                        allEdges_f[connectedNodes[0]][connectedNodes[1]] += 1
                    else:
                        allEdges_m[connectedNodes[0]][connectedNodes[1]] += 1
                
                
                edgeVal = [allEdges_f, allEdges_m]
                
                insights.append({
                    "value" : edgeVal,
                    "name" : "subgraph_edges",
                    "type" : "array"
                })
                    




            insights.append({
                "value" : value,
                "name" : key,
                "type" : "array"
            })

        return insights


