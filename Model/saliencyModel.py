
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F 
from Model.windower import calWindowedFcMatrixes, getGraphicalData_fromFcMatrixes, calFcMatrixes, windowBoldSignal
from Model.DistanceMatrix.getDistanceEdges import getDistanceEdges, getDistanceMatrix
from einops import repeat, rearrange

class Model():

    def __init__(self, mainOptions, model):

        self.mainOptions = mainOptions

        self.model = model.to(mainOptions.device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
       

    
    def step(self, x, y):

        """
            x = (batchSize, N, dynamicLength) 
            y = (batchSize, numberOfClasses)
        
        """
        self.model.eval()

        wBOLD = self.getWindowBolds(x)

        inputs, y = self.prepareInput(x, y)

        yHat, _ = self.model(*inputs)

        pred = np.argmax(yHat.cpu().data.numpy(), axis=-1)
        

        # accumulate gradients on attentions
        one_hot = np.zeros((1, yHat.size()[-1]), dtype=np.float32)
        one_hot[0, pred] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot = torch.sum(one_hot.to(self.mainOptions.device) * yHat)
        self.model.zero_grad()
        one_hot.backward()

        if(self.mainOptions.useGraphCorr):
            input_grads = inputs[7].grad.data.abs() # prior connectivity matrix (windowed)
            # grads.shape: 19, 400, 400 (19 windows)
            # input_grads = torch.mean(input_grads, dim = 0) # average over windows
            
        else:
            input_grads = inputs[0].grad.data.abs() # prior connectivity matrix (not windowed)
            # input_grads.shape: 400,400
        
        grads = input_grads.to("cpu")

        return grads, pred, wBOLD            

    def getWindowBolds(self, x):
        T = x.shape[2]
        
        windowedX, _ = windowBoldSignal(x, T, self.mainOptions.windowSize, self.mainOptions.stride)
        return windowedX

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
        #### assuming corr + sage or corr only

        batchSize = x.shape[0]
        N = x.shape[1]
        T = x.shape[2]
        
        windowedX, _ = windowBoldSignal(x, T, self.mainOptions.windowSize, self.mainOptions.stride)
        # windowedX shape = (batchSize, #windows, N, windowLength)
        windowedFC = calWindowedFcMatrixes(windowedX) # of shape = (batchSize, #windows, N, N)

        extX = x.unsqueeze(dim = 1) # batchSize, 1, N, T
        fcMatrixes = calWindowedFcMatrixes(extX) # of shape = (batchSize, 1, N, N)
        windowCount = fcMatrixes.shape[1] # 1 window at start ( 1 x batch graphs will be generated)

        graphicalData_graph = getGraphicalData_fromFcMatrixes(fcMatrixes, fcMatrixes, self.mainOptions.sparsity_graph, "value_thresholded")
        
        distanceMatrix = getDistanceMatrix() # (N, N)
        distanceMatrix = torch.tensor(repeat(distanceMatrix, "n m -> b w n m", b = batchSize, w = windowCount))
        graphicalData_corr = getGraphicalData_fromFcMatrixes(distanceMatrix, extX, self.mainOptions.sparsity_corr, "connection_thresholded")

        # load necessaries to gpu
        if(self.mainOptions.useGraphCorr):
            x_corr = graphicalData_corr.x.to(self.mainOptions.device)
        

            edge_index_corr = graphicalData_corr.edge_index.to(self.mainOptions.device)
            prior_connectivityMatrix = rearrange(windowedFC, "b w n m -> (b w) n m").to(self.mainOptions.device)
            prior_connectivityMatrix = Variable (prior_connectivityMatrix, requires_grad = True)

            edge_index_graph = graphicalData_graph.edge_index.to(self.mainOptions.device)
        
            batch = graphicalData_graph.batch.to(self.mainOptions.device) # does not matter where it comes from, graphicalData_corr is also same
            y = y.to(self.mainOptions.device)


            return (
                None, # x_graph
                edge_index_graph,
                batch,

                None,
                None,

                x_corr,
                edge_index_corr,
                prior_connectivityMatrix

            ), y 
        
        else:
            # sage only
            x_graph = graphicalData_graph.x.to(self.mainOptions.device) # node features as FCs
            x_graph = Variable(x_graph, requires_grad = True)
            edge_index_graph = graphicalData_graph.edge_index.to(self.mainOptions.device)
            batch = graphicalData_graph.batch.to(self.mainOptions.device) # does not matter where it comes from, graphicalData_corr is also same
            
            return (
                x_graph,
                edge_index_graph,
                batch,

                None,
                None,

                None,
                None,
                None

            ), y 
        

    def getLoss(self, yHat, y):
        
        cross_entropy_loss = self.criterion(yHat, y)

        return cross_entropy_loss



    
