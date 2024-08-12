import numpy as np
import torch_geometric
import torch

global fc
global edge_index

fc = None
edge_index = None

def getDistanceMatrix():
    global fc

    if(not isinstance(fc, type(None))):
        return fc

    fc = np.load("./Model/DistanceMatrix/distanceMatrix.npy")

    return fc

def getDistanceEdges(sparsity, batchSize):

    global fc
    global edge_index

    

    if(not isinstance(fc, type(None))):
        N = fc.shape[0]
        return broadcastEdges_toBatches(edge_index, N, batchSize)

    fc = np.load("./Model/DistanceMatrix/distanceMatrix.npy")
    N = fc.shape[0]

    numberOfConnectionsPerRoi = int(N * sparsity / 100) 
    deadIndexes = np.argsort(np.abs(fc), axis=1)[:, :-numberOfConnectionsPerRoi]
    deadIndexes += np.repeat(np.arange(N).reshape(-1,1) * N, repeats=deadIndexes.shape[-1], axis=-1)

    fc = fc.reshape(-1)
    fc[deadIndexes.flatten()] = 0
    fc = fc.reshape(N, N)        

    edge_index = torch_geometric.utils.dense_to_sparse(torch.tensor(fc))[0]
    #remove self loops
    edge_index = torch_geometric.utils.remove_self_loops(edge_index)[0]

    return broadcastEdges_toBatches(edge_index, N, batchSize)

def broadcastEdges_toBatches(edges, maxNodeIndex, batchSize):
    """
        edges : (2, original_edgeCount),
        maxNodeIndex : Number,
        batchsize : Number
    """

    singleGraph_edgeCount = edges.shape[1]
    edges_broadCasted = edges.tile((1, batchSize))
    edges_broadCasted += torch.repeat_interleave((torch.arange(batchSize) * maxNodeIndex).unsqueeze(dim=0), singleGraph_edgeCount, dim=-1)
    
    return edges_broadCasted