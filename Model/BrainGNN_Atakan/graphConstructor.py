import torch
from einops import rearrange, repeat
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric
import numpy as np

import time

from nilearn.connectome import ConnectivityMeasure

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
        fcMatrixes   = (batchSize, N, N)
        nodeFeatures = (batchSize, N, F)
        sparsity = Number between 0 and 100, 0 means sparse, 100 means fully connected
    """

    batchSize = fcMatrixes.shape[0]
    N = fcMatrixes.shape[2] # number of nodes


    graphDatas = []

    for batch in range(batchSize):
            
        time_start = time.time()

        FC = fcMatrixes[batch]

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
            
        x = torch.tensor(nodeFeatures[batch])

        pos = torch.tensor(np.eye(FC.shape[0])).float()

        graphData = Data(x=x, edge_index = edge_index, edge_attr = edge_attr.unsqueeze(dim=-1), pos=pos)
        graphDatas.append(graphData)
            
        time_data = time.time()
            
    #print("thresholding : {}, edgeConsruct : {}".format(time_thresholding-time_start, time_edge-time_thresholding))

    # here constructing big disconnected graph by leveraging torch_geometric function
    allGraphicalData = next(iter(DataLoader(graphDatas, batch_size = batchSize))) # Data(x, edge_index)

    return allGraphicalData
    
