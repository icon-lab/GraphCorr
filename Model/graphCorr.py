import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

from einops import rearrange, repeat
from torch import nn
from torch_geometric.nn.inits import glorot

from .windower import windowNodeSignal


class AttentionEmbedder(nn.Module):

    def __init__(self, N, D, heads=8, dimHead=50):

        super().__init__()

        self.N = N
        self.D = D
        self.heads = heads

        innerDim = dimHead * heads

        self.scale = dimHead ** -0.5

        self.layerNorm_attn = nn.LayerNorm(N)
        self.to_qkv = nn.Linear(N, 3*innerDim, bias = False) #

        self.attend = nn.Softmax(dim=-1)

        self.attnOut = nn.Sequential(
            nn.Linear(innerDim, N),
            nn.Dropout(0.5)
        )

        self.mlp = nn.Sequential(
            nn.Linear(N, D),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(D, D),
            nn.Dropout(0.5)
        )


        self.layerNorm_mlp = nn.LayerNorm(N)


        

    def forward(self, priorMatrix):

        """
            priorMatrix : (batchSize*windowCount, N, N)
        """
        priorMatrix = self.layerNorm_attn(priorMatrix) # (batchSize*w, N, N)
        
        qkv = self.to_qkv(priorMatrix).chunk(3, dim=-1) # [(batchSize, N, innerDim) * 3]
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        
        # each shape of q k v = (batchSize*w, h, N, dimHead)
        dots = torch.matmul(q, k.transpose(-1,-2)) * self.scale # (batchSize, h, N, N)
        attn = self.attend(dots) # (batchSize, N, N)
        attended = torch.matmul(attn, v) # (batchSize, h, N, dimHead)
        attended = rearrange(attended, "b h n d -> b n (h d)") # (batchSize, N, innerDim)
        attnOut = self.attnOut(attended) # (batchSize, N, N)

        mlpOut = self.mlp(self.layerNorm_mlp(attnOut + priorMatrix)) # (batchSize*w, N, D)
  

        return mlpOut

    

        

class GraphCorr(MessagePassing):

    def __init__(self, numberOfNodes=400, learnedEmbedDim = 50,  maxLag=5, k=3, inductiveBias=True):

        super().__init__(aggr='add') #
        
        self.numberOfNodes = numberOfNodes
        self.learnedEmbedDim = learnedEmbedDim
        self.maxLag = maxLag
        self.k = k

        
        self.inductiveBias = inductiveBias

        self.embedder = AttentionEmbedder(N=numberOfNodes, D=learnedEmbedDim)
        self.Wc = nn.Parameter(torch.randn(k, 1 + 2 * (maxLag)), requires_grad=True)
        self.nodeNorm = nn.LayerNorm(learnedEmbedDim * k)
        self.nodeNorm2 = nn.LayerNorm(numberOfNodes * k)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        with torch.no_grad():

            if(self.Wc.shape[0] == self.Wc.shape[1] and False):
                self.Wc += torch.diag(torch.ones(self.Wc.shape[0])) * (5 * torch.max(torch.abs(self.Wc)))

            if(self.Wc.shape[1] * self.Wc.shape[0] > 0):
                glorot(self.Wc)
                self.Wc /= 10000

                if(self.inductiveBias):

                    if(self.Wc.shape[0] > self.Wc.shape[1]):
                        pass
                    else:
                        pass
                
        
        #glorot(self.Wc)
        
    def forward(self, useTransformer, useLSTM, x, priorMatrix, edge_index, dynamicLength, windowSize, stride):

        """  
            x : (Ne, T)
            priorMatrix : (batchSize, N, N)
            edge_index : (2, Ni)
        """

        windowCount = (dynamicLength-windowSize) // stride
        
        batchSize = priorMatrix.shape[0]
        
        # priorMatrix.shape: (batchSize*w, N, N)
        
        if useTransformer:
            nodeEmbeddings = self.embedder(priorMatrix) # (batchSize*w, N, D)

        elif(dynamicLength == windowSize):
            nodeEmbeddings = priorMatrix

        else:
            nodeEmbeddings = priorMatrix 
        newNodeFeatures = self.propagate(edge_index=edge_index, x=x, nodeEmbeddings=nodeEmbeddings, dynamicLength=dynamicLength, windowSize=windowSize, stride=stride, useLSTM = useLSTM) # ((batchSize * N), D*K)

        if useTransformer:
            newNodeFeatures = self.nodeNorm(newNodeFeatures) #((batchSize*N), D*K)
        else:
            # normalize the features
            newNodeFeatures = self.nodeNorm2(newNodeFeatures) # ((batchSize*N), N*K)

        return newNodeFeatures, nodeEmbeddings


    def message(self, x_i, x_j, edge_index, nodeEmbeddings, dynamicLength, windowSize, stride, useLSTM):

        """
            x_ : (Ne, T)
            edge_index : (2, Ne)
            nodeEmbeddings : (batchSize*w, N, D)
        """
        
        Ne = x_i.shape[0]
        

        # get windowed node bold sequences
        windowedXi, _ = windowNodeSignal(x_i, dynamicLength, windowSize, stride) # Ne, window count, window length
        windowedXj, _ = windowNodeSignal(x_j, dynamicLength, windowSize, stride)

        
        w = windowedXi.shape[1]

        if useLSTM:
            
            windowedXi = rearrange(windowedXi, "n w t -> t (n w)").unsqueeze(dim=-1) # window length, Ne*window count, 1
            windowedXj = rearrange(windowedXj, "n w t -> t (n w)").unsqueeze(dim=-1) # window length, Ne*window count, 1
            lstmIn = torch.cat([windowedXi,windowedXj], dim=-1)
            lstmOut, _ = self.lstm(lstmIn) # window length, Ne*window count, k
            lstmOut = torch.mean(lstmOut, dim=0) # Ne*window count, k
            filterValues = lstmOut.unsqueeze(dim=1) # (Ne*w, 1, K)
        else:
            # get them ready for convolution
            windowedXj = rearrange(windowedXj, "n w t -> (n w) t").unsqueeze(dim=0) # 1, Ne*window count, window length
            windowedXi = rearrange(windowedXi, "n w t -> (n w) t").unsqueeze(dim=1) # Ne*window count, 1, window length

            # calculate correlations between the windowed bold patches
            corr = F.conv1d(windowedXj, windowedXi, padding=self.maxLag, groups=Ne*w).view(w*Ne,-1) #/ (nonzero_dim**0.5) # of shape (Ne*w, 2*padding + 1)
            corrExt = rearrange(corr, "(n w) p -> n w p", n = Ne)
            
            filterValues = F.gelu(torch.matmul(corrExt, self.Wc.T)) # (Ne, w, K)
            #filterValues = torch.mean(filterValues, dim=1) # of shape (Ne, K)
            
            # now broadcast filtered values to corresponding learned node embeddings
            filterValues = rearrange(filterValues, "n w k -> (n w) k").unsqueeze(dim=1) # (Ne*w, 1, K)
        
        nodeEmbeddings = rearrange(nodeEmbeddings, "(b w) n d -> b w n d", w=w)
        nodeEmbeddings = rearrange(nodeEmbeddings, "b w n d -> b n w d")
        nodeEmbeddings = rearrange(nodeEmbeddings, "b n w d -> (b n) (w d)") # batch*N, w*D
        
        indexedEmbeddings = nodeEmbeddings[edge_index[1]]
        indexedEmbeddings = rearrange(indexedEmbeddings,"n (w d) -> (n w) d", w=w).unsqueeze(dim=-1) # (Ne*w, D, 1)


        broadCastedFilterValues = torch.matmul(indexedEmbeddings, filterValues) # (Ne*w, D, K)
        broadCastedFilterValues = rearrange(broadCastedFilterValues, "(n w) d k -> n w d k", w=w, n=Ne)
        broadCastedFilterValues = torch.mean(broadCastedFilterValues, dim=1)
        out = rearrange(broadCastedFilterValues, "n d k -> n (d k)") # (Ne, D*K)

        # out shape is Ne, N*K if transformer is not used
        
        return out 
        
