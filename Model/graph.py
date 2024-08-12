
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TAGConv, GCNConv
import torch.nn as nn

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, TopKPooling as topk

from .graphCorr import GraphCorr
from einops import rearrange
from Model.BrainGNN_Atakan.brainGnn import BrainGnn
from Model.BrainNetCnn_Atakan.brainNetCnn import BrainNetCNN

from .parcelGraph import parcelForward_graph, concatParcellatedOuts_corrConv


graphNameToLayer = {
    "sage" : SAGEConv,
    "tag" : TAGConv,
    "gcn" : GCNConv
}

poolingNameToFunc = {
    "gmp" : gmp,
    "gap": gap
}

class GraphNN(torch.nn.Module):

    def __init__(self, mainOptions):

        super().__init__()

        useGraphCorr = mainOptions.useGraphCorr
        

        nOfClasses = mainOptions.nOfClasses
        dropout = mainOptions.dropout
        nOfRois = mainOptions.nOfRois
        graphDim = mainOptions.graphDim
        windowCount = (mainOptions.dynamicLength-mainOptions.windowSize) // mainOptions.stride + 1

        if mainOptions.useGraphCorr:
            if mainOptions.useTransformer:
                graphInDim = mainOptions.learnedEmbedDim * (mainOptions.k + 1)
            else:
                graphInDim = nOfRois * (mainOptions.k + 1)
        
        else:
            graphInDim = nOfRois
            

        self.mainOptions = mainOptions
        
        
        if(useGraphCorr):

            learnedEmbedDim = mainOptions.learnedEmbedDim
            maxLag = mainOptions.maxLag
            k = mainOptions.k
            
            self.graphCorr = GraphCorr(nOfRois, learnedEmbedDim, maxLag, k)


        if (mainOptions.graphLayer == "brainGnn"):
            self.graphLayers = BrainGnn(mainOptions)
        elif (mainOptions.graphLayer == "brainNetCnn"):
            self.graphLayers = BrainNetCNN(mainOptions)
        else:
            graphLayer = graphNameToLayer[mainOptions.graphLayer]
            
            graphLayers = [graphLayer(graphInDim, graphDim)]
            print(graphInDim)
            for i in range(mainOptions.nOfLayers-1):
                graphLayers.append(graphLayer(graphDim, graphDim))

            self.graphLayers = nn.ModuleList(graphLayers)

        self.pool = poolingNameToFunc[mainOptions.poolingMethod]



        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(graphDim, nOfClasses)

    def forward(self, x_graph, edge_index_graph, batch, edge_attr, pos, x_corr=None, edge_index_corr = None, prior_connectivityMatrix=None):

        """

            x_graph: (batchSize * nodeCount, N)
            if stat+dyn FC used => x_graph: (batchSize * (windowCount+1) * N, N)
            edge_index_graph: (2, (batchSize * nodeCount * #ofConnectionsPerRoi))
            batch : ??      
                        
            x_corr : (batchSize * windowCount * nodeCount, Tw)
            edge_index_corr : (2, (batchSize * windowCount * nodeCount * #ofConnectionsPerRoi))
            prior_connectivityMatrix = (batchSize * windowCount, nodeCount, nodeCount)


            Note:
            
                if(useGraphCorr):

                    x_graph is not used

                else:

                    x_corr, edge_index_corr, prior_connectivityMatrix are not used

        """

        
        if(self.mainOptions.useGraphCorr):


            nodeCount = prior_connectivityMatrix.shape[-1]
            
            batchSize = x_corr.shape[0] // nodeCount 

            
            newNodeFeatures, nodeEmbeddings = self.graphCorr(self.mainOptions.useTransformer, self.mainOptions.useLSTM, x_corr, prior_connectivityMatrix, edge_index_corr,self.mainOptions.dynamicLength, self.mainOptions.windowSize, self.mainOptions.stride)
            

            nodeEmbeddings = rearrange(nodeEmbeddings, "(b w) n d -> b w n d", b=batchSize)
            nodeEmbeddings = torch.mean(nodeEmbeddings, dim=1)

            embedFeatures = rearrange(nodeEmbeddings, "b n d -> (b n) d")

            

            if(self.mainOptions.k > 0):
                x = torch.cat([newNodeFeatures, embedFeatures], dim=1)
                
            else:
                x = embedFeatures

            if(self.mainOptions.graphLayer == "brainNetCnn"): 
                               
                x = rearrange(x, "(b n) f -> b n f", b=batchSize).unsqueeze(dim=1)
                
        else:
            nodeCount = x_graph.shape[-1]
            batchSize = x_graph.shape[0] // nodeCount
            x = x_graph
            print(x.shape)

        
        if (self.mainOptions.graphLayer == "brainGnn"):

            logits = self.graphLayers(x_graph=x, edge_index=edge_index_graph, batch=batch, edge_attr=edge_attr, pos=pos)
            
        elif (self.mainOptions.graphLayer == "brainNetCnn"):
            
            logits = self.graphLayers(x)
        else:
            for i in range(self.mainOptions.nOfLayers):
                x = self.graphLayers[i](x, edge_index_graph)
 
                if(i != self.mainOptions.nOfLayers):
                    x = self.dropout(F.relu(x))

            # x.shape = (batchSize * N, finalFeatureDim)

            # pool inside the graphs


            pooledX = self.pool(x, batch) # (batchSize, finalFeatureDim)
            
            # classification
            logits = self.classifier(self.dropout(F.relu(pooledX))) # (batchSize, #ofClasses)


        lagMatrix = None #if not self.mainOptions.useGraphCorr else self.graphCorr.Wc.detach().cpu().numpy()
        
        return logits, lagMatrix
