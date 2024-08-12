import torch
from einops import rearrange, repeat
from Utils.gpuChecker import getGpuLoad

def parcelForward_graph(forwardGnn, x, priorMatrix, edge_index, batchIndexes, nOfParcels, useTransformer, useLSTM, dynamicLength, windowSize, stride):
    """
        Input:
            forwardGnn : gnn model to be used
            x: (N, H), N : number of nodes in the batch graph
            M : (N, ...) : Optional extra input matrix to the graph layer
            edge_index : (2, Ne), Ne : number of edges in the batch graph
            batchIndexes : (N,)
        Output:
            results : an list of len nOfParcels, it contains outputs of forwardGnn        
        Assumptions:
            * edge_indexes are sorted along first dim
    """

    batchSize = torch.max(batchIndexes) + 1
    M = rearrange(priorMatrix, "(b w) n m -> b w n m", b=batchSize)


    if(batchSize % nOfParcels != 0):
        print("Batchsize is not integer divisible with nOfParcels, making nOfParcels = batchsize")
        nOfParcels = batchSize

    results = []

    for i in range(nOfParcels):

        sliceLeft = int(len(batchIndexes) / nOfParcels * i)
        sliceRight = int(len(batchIndexes) / nOfParcels * (i+1))

        x_parcel = x[sliceLeft : sliceRight]
        
        edge_index_parcel = edge_index[:, torch.logical_and(edge_index[0] >= sliceLeft, edge_index[0] < sliceRight) ] - sliceLeft # removing the min node index

        M_parcel = M[batchIndexes[sliceLeft] : batchIndexes[sliceRight-1] + 1] # (sliced b, w, n, n)
        M_parcel = rearrange(M_parcel, "sb w n m -> (sb w) n m")
        

       

        nF_parcel, nE_parcel = forwardGnn(useTransformer, useLSTM, x_parcel, M_parcel, edge_index_parcel,dynamicLength, windowSize, stride)
        
        

        out_parcel = (nF_parcel.to("cpu"), nE_parcel.to("cpu"))
        torch.cuda.empty_cache()

        results.append(out_parcel)

        print(i, "gpuLoad = {}".format(getGpuLoad(0)))



    return results


def concatParcellatedOuts_corrConv(results):

    nodeFeatures = []
    nodeEmbeddings = []

    for out in results:
        nodeFeatures.append(out[0])
        nodeEmbeddings.append(out[1])
    
    return torch.cat(nodeFeatures, dim=0), torch.cat(nodeEmbeddings, dim=0)

def concatParcellatedOuts_gin(results):

    return torch.cat(results, dim=0)