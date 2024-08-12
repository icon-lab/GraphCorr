import torch
import numpy as np

import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch.backends.cudnn as cudnn



class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, numberOfNodes, inputDim, bias=False):
        super(E2EBlock, self).__init__()
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, inputDim), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (numberOfNodes, 1), bias=bias)
        self.numberOfNodes = numberOfNodes
        
        
    def forward(self, x):
        inputDim = x.shape[-1]
        a = self.cnn1(x)
        
        b = self.cnn2(x)
        out = torch.cat([a]*inputDim,3)+torch.cat([b]*self.numberOfNodes,2)
        
        return out
    
    
class BrainNetCNN(torch.nn.Module):
    def __init__(self, mainOptions):
        super(BrainNetCNN, self).__init__()

        self.mainOptions = mainOptions
        self.dropout = mainOptions.dropout

        hiddenDim = mainOptions.graphDim

        if mainOptions.useGraphCorr:
            inputDim = mainOptions.learnedEmbedDim * (mainOptions.k + 1)
        else:
            inputDim = mainOptions.nOfRois
        
        self.e2econv1 = E2EBlock(1,hiddenDim, mainOptions.nOfRois, inputDim, bias=True)
        self.e2econv2 = E2EBlock(hiddenDim, hiddenDim*2, mainOptions.nOfRois, inputDim, bias=True)
        self.E2N = nn.Conv2d(hiddenDim*2, 1,(1,inputDim))
        self.N2G = nn.Conv2d(1, hiddenDim*8,(mainOptions.nOfRois,1))
        self.dense1 = nn.Linear(hiddenDim*8, hiddenDim*4)
        self.bN1_1d = torch.nn.BatchNorm1d(hiddenDim*4)
        self.dense2 = nn.Linear(hiddenDim*4,hiddenDim)
        self.dense3 = nn.Linear(hiddenDim,mainOptions.nOfClasses)

        
    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x),negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out),negative_slope=0.33) 
        out = F.leaky_relu(self.E2N(out),negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out),negative_slope=0.33) , p=self.dropout) 
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.bN1_1d(self.dense1(out)),negative_slope=0.33) , p=self.dropout)
        out = F.dropout(F.leaky_relu(self.dense2(out),negative_slope=0.33) , p=self.dropout)
        out = self.dense3(out) # removed leak relu, since our goal is classification not regression
        
        return out
    
    