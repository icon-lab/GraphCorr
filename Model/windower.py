import torch
from einops import rearrange, repeat
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric
import numpy as np

import time

from nilearn.connectome import ConnectivityMeasure

def windowBoldSignal(boldSignal, dynamicLength, windowLength, stride, randomSeed=None):
    
    """
        boldSignal : (batchSize, N, T)

        output : (batchSize, (T-windowLength) // stride, N, windowLength )

    """

    if(randomSeed != None):
        torch.manual_seed(randomSeed)
    
    T = boldSignal.shape[2]
    # NOW WINDOWING 
    windowedBoldSignals = []
    samplingEndPoints = []

    for windowIndex in range((T - windowLength)//stride + 1):
        
        sampledWindow = boldSignal[:, :, windowIndex * stride  : windowIndex * stride + windowLength]
        samplingEndPoints.append(windowIndex * stride + windowLength)

        sampledWindow = torch.unsqueeze(sampledWindow, dim=1)
        windowedBoldSignals.append(sampledWindow)
        
    windowedBoldSignals = torch.cat(windowedBoldSignals, dim=1)
    
    return windowedBoldSignals, samplingEndPoints

def windowNodeSignal(boldSignal, dynamicLength, windowLength, stride, randomSeed=None):
    
    """
         node boldSignal : (Ne, T)

        output : (Ne, (T-windowLength) // stride, windowLength )

    """

    if(randomSeed != None):
        torch.manual_seed(randomSeed)
    
    T = boldSignal.shape[1]
    # NOW WINDOWING 
    windowedBoldSignals = []
    samplingEndPoints = []

    for windowIndex in range((T - windowLength)//stride + 1):
        
        sampledWindow = boldSignal[:, windowIndex * stride  : windowIndex * stride + windowLength] # Ne, windowLength
        samplingEndPoints.append(windowIndex * stride + windowLength)

        sampledWindow = torch.unsqueeze(sampledWindow, dim=1) # Ne, 1, window length
        windowedBoldSignals.append(sampledWindow)
        
    windowedBoldSignals = torch.cat(windowedBoldSignals, dim=1) # Ne, window count, window length

    return windowedBoldSignals, samplingEndPoints
# corrcoef based on
# https://github.com/pytorch/pytorch/issues/1254

def calWindowedFcMatrixes(x):

    """
        x   = (batchSize, #windows, N, T)
        out = (batchSize, #windows, N, N)
    """

    batchSize = x.shape[0]
    numberOfWindows = x.shape[1]
    N = x.shape[2]

    mean_x = torch.mean(x, -1, keepdim=True)
    xm = x.sub(mean_x)
    
    # reshape
    xm = rearrange(xm, "b w ... -> (b w) ...")

    # node variance calc
    c = xm.bmm(torch.transpose(xm, dim0=1, dim1=2))
    c = c / (x.size(3) - 1)
        
    d = torch.diagonal(c, dim1=1, dim2=2)
    stddev = torch.unsqueeze(torch.pow(d, 0.5), dim=-1)
    
    # normalize by node stds
    c = c.div(stddev.expand_as(c))    
    c = c.div(torch.transpose(stddev.expand_as(c), dim0=1, dim1=2))
    c = torch.clamp(c, -1.0, 1.0)

    # reshape
    c = rearrange(c, "(b w) ... -> b w ...", b = batchSize)
    

    return c

def calFcMatrixes(x, correlationType):

    """
        Input:
            x = (batchSize, T, N)
            correlationType = one of ["correlation", "partial correlation"]
        Output:
        out = (batchSize, N, N)
    """
    
    correlationMeasure = ConnectivityMeasure(kind=correlationType)

    return correlationMeasure.fit_transform(x)

def getGraphicalData_fromFcMatrixes(fcMatrixes, nodeFeatures, sparsity, edgeMethod):
    
    """
        fcMatrixes   = (batchSize, 1, N, N)
        nodeFeatures = (batchSize, 1, N, F)
        sparsity = Number between 0 and 100, 0 means sparse, 100 means fully connected
    """

    batchSize = fcMatrixes.shape[0]
    N = fcMatrixes.shape[-1] # number of nodes

    fcMatrixes = torch.squeeze(fcMatrixes, dim=1)
    #print(fcMatrixes.shape)
    nodeFeatures = torch.squeeze(nodeFeatures, dim=1)
    #print(nodeFeatures.shape)
    graphDatas = []

    for batch in range(batchSize):
            
        time_start = time.time()

        FC = fcMatrixes[batch].numpy()

        if(edgeMethod == "value_thresholded"):
            
            FC = FC > np.percentile(FC, 100 - sparsity)
            
        elif(edgeMethod == "connection_thresholded"):
                
            numberOfConnectionsPerRoi = int(N * sparsity / 100) 
            deadIndexes = np.argsort(np.abs(FC), axis=1)[:, :-numberOfConnectionsPerRoi-1]
            deadIndexes += np.repeat(np.arange(N).reshape(-1,1) * N, repeats=deadIndexes.shape[-1], axis=-1)

            FC = FC.reshape(-1)
            FC[deadIndexes.flatten()] = 0
            FC = FC.reshape(N, N)


        time_thresholding = time.time()

        edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(torch.tensor(FC))
            
        #remove self loops
        edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)

        time_edge = time.time()
            
        x = nodeFeatures[batch]

        pos = torch.tensor(np.eye(FC.shape[0])).float()

        graphData = Data(x=x, edge_index = edge_index, edge_attr = edge_attr.unsqueeze(dim=-1), pos=pos)
        graphDatas.append(graphData)
            
        time_data = time.time()
            
    #print("thresholding : {}, edgeConsruct : {}".format(time_thresholding-time_start, time_edge-time_thresholding))

    # here constructing big disconnected graph by leveraging torch_geometric function
    allGraphicalData = next(iter(DataLoader(graphDatas, batch_size = batchSize))) # Data(x, edge_index)

    return allGraphicalData
    


def calGlobalFcMatrix(x):

    """
        x   = (batchSize, N, T)
        out = (batchSize, N, N)
    """

    batchSize = x.shape[0]
    
    N = x.shape[1]

    mean_x = torch.mean(x, -1, keepdim=True)
    xm = x.sub(mean_x)
    

    # node variance calc
    c = xm.bmm(torch.transpose(xm, dim0=1, dim1=2))
    c = c / (x.size(2) - 1)
        
    d = torch.diagonal(c, dim1=1, dim2=2)
    stddev = torch.unsqueeze(torch.pow(d, 0.5), dim=-1)
    
    # normalize by node stds
    c = c.div(stddev.expand_as(c))    
    c = c.div(torch.transpose(stddev.expand_as(c), dim0=1, dim1=2))
    c = torch.clamp(c, -1.0, 1.0)
    

    return c
    
    