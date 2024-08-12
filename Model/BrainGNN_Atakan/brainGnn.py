import torch
from torch import nn

from .net.braingnn import Network


class BrainGnn(nn.Module):

    def __init__(self, mainOptions):

        super().__init__()

        if mainOptions.useGraphCorr:
            inputDim = mainOptions.learnedEmbedDim * (mainOptions.k + 1)
        else:
            inputDim = mainOptions.nOfRois
        
        poolingRatio = mainOptions.brainGnn.poolingRatio
        nOfClasses = mainOptions.nOfClasses
        k = 8 # number of communities

        self.network = Network(inputDim, poolingRatio, nOfClasses, k, mainOptions.nOfRois)

    def forward(self, x_graph, edge_index, edge_attr, batch, pos):

        logits = self.network( x=x_graph, edge_index=edge_index, batch=batch, edge_attr=edge_attr, pos=pos)

        return logits
