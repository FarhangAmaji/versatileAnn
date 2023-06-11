#%% imports
# models\multivariateTransformers.py
import sys
import os
parentFolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentFolder)
from versatileAnn import ann
import torch
from torch import nn
#%% define model
class TransformerInfo:
    def __init__(self, embedSize=32, heads=8, forwardExpansion=4, encoderLayersNum=6, decoderLayersNum=6, dropoutRate=.6, inpLen=10, outputLen=10, inputDim=1, outputDim=1):
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedSize = embedSize
        self.heads = heads
        self.forwardExpansion = forwardExpansion
        self.dropoutRate = dropoutRate
        self.headDim = embedSize // heads
        assert (self.headDim * heads == embedSize), "Embedding size needs to be divisible by heads number"

        self.inpLen= inpLen
        self.outputLen= outputLen
        self.inputDim= inputDim
        self.outputDim= outputDim
        self.encoderLayersNum= encoderLayersNum
        self.decoderLayersNum= decoderLayersNum
        self.encoderPositionalEmbedding=self.positionalEmbedding2d(inpLen, inputDim)
        self.decoderPositionalEmbedding=self.positionalEmbedding2d(outputLen+1, outputDim)#+1 is for last data from input added to the first of output
        '#ccc for regression data we dont have startToken like in nlp so we assume 0 of same shape of input is the first output; therefore we add +1 to outputLen'
        self.trgMask=self.makeTrgMask(outputLen+1)#jjj
    
    def makeTrgMask(self, outputLen):
        trgMask = torch.tril(torch.ones((outputLen, outputLen)))
        return trgMask.to(self.device)
    
    def positionalEmbedding2d(self, maxRows, maxCols):#jjj add shapes and comment
        positionRows = (torch.arange(maxRows).reshape(maxRows, 1).repeat(1, maxCols))
        positionEnc = torch.zeros(maxRows, maxCols, self.embedSize)
        # Calculate positional encodings for rows and columns
        for d in range(0, self.embedSize, 2):
            denominator = torch.pow(10000.0, torch.tensor(d, dtype=torch.float32) / self.embedSize)
            positionEnc[:, :, d] = torch.sin(positionRows / denominator)
            positionEnc[:, :, d + 1] = torch.cos(positionRows / denominator)
        return positionEnc.to(self.device)
    
class multiHeadAttention(nn.Module):
    def __init__(self, transformerInfo, queryDim, valueDim):#jjj correct comments
        super(multiHeadAttention, self).__init__()
        '''#ccc its important to know the network is independent of sequence length because we didnt make any layer(linear)
        which takes input equal to sequence length
        '''
        self.transformerInfo= transformerInfo
        self.valueLayer = nn.Linear(valueDim * transformerInfo.embedSize, valueDim * transformerInfo.embedSize)
        self.keyLayer = nn.Linear(valueDim * transformerInfo.embedSize, valueDim * transformerInfo.embedSize)
        self.queryLayer = nn.Linear(queryDim * transformerInfo.embedSize, queryDim * transformerInfo.embedSize)
        self.outLayer = nn.Linear(queryDim * transformerInfo.embedSize, queryDim * transformerInfo.embedSize)#jjj

    def forward(self, query, key, value, mask):
        'multiHeadAttention'
        'this can be used for encoder and both decoder attentions'
        '#ccc in 2nd attention in decoder, embedded output is query'
        # Get batchSize
        N = query.shape[0]

        valueLen, queryLen = value.shape[1], query.shape[1]
        valueDim, queryDim= value.shape[2], query.shape[2]
        """#ccc the valueLen/valueDim and keyLen/keyDim in both attentions in decoder are the same
        but the queryLen/queryDim in 1st attention in decoder is same as them and in 2nd attention not the same"""
        value = self.valueLayer(value.reshape(N, valueLen, -1))  # N * valueLen * valueDim * embedSize
        key = self.keyLayer(key.reshape(N, valueLen, -1))  # N * valueLen * valueDim * embedSize
        query = self.queryLayer(query.reshape(N, queryLen, -1))  # N * queryLen * queryDim * embedSize

        # Split the embedding into heads different pieces
        value = value.reshape(N, valueLen, valueDim, self.transformerInfo.heads, self.transformerInfo.headDim)#jjj
        key = key.reshape(N, valueLen, valueDim, self.transformerInfo.heads, self.transformerInfo.headDim)
        query = query.reshape(N, queryLen, queryDim, self.transformerInfo.heads, self.transformerInfo.headDim)

        energy = torch.einsum("nqxhd,nkyhd->nhqxky", [query, key])#jjj#kkk
        'qxd,kyd->qxky'
        'qxky=qx1*ky1+qx2*ky2+...'
        """#ccc Einsum does matrix multiplication
        the output dimension is n*h*queryLen*valueLen
        attentionTable(queryLen*valueLen) which for each row(word), shows the effect other words(each column)
        the queryLen in 2nd attention is the output len
        the valueLen in 2nd attention is the input len
        
        n and h exist in nhqv,nvhd and nqhd. so we can think qd,kd->qk
        so for a specific q and a specific k(in result tensor): for each d we have multiplied "the dth embedding value of qd" to "the dth embedding value of kd", then summed them up
        qk=q1*k1+q2*k2+...
        so each element in qk is dot product of qth word and kth word"""
        # query shape: N * queryLen * heads * heads_dim
        # key shape: N * valueLen * heads * heads_dim
        # energy: N * heads * queryLen * valueLen

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))#jjj mask for decoder
        
        energy=energy.reshape(N, self.transformerInfo.heads, queryLen, queryDim, -1)
        attention = torch.softmax(energy / (self.transformerInfo.embedSize ** (1 / 2)), dim=4)#jjj#kkk
        attention=attention.reshape(N, self.transformerInfo.heads, queryLen, queryDim, valueLen, valueDim)
        # attention shape: N * heads * queryLen * valueLen
        '''#ccc we Normalize energy value for better stability
        note after this we would have attention[0,0,0,:].sum()==1
        
        note one question comes to mind "we are doing softmax, what difference would deviding by
        'embedSize ** (1 / 2)' make,finally we are doing softmax":? yes because softmax does averaging exponentially so smaller
        value get so less softmax output
        '''
        out = torch.einsum("nhqxvy,nvyhd->nqxhd", [attention, value]).reshape(N, queryLen, queryDim, self.transformerInfo.heads * self.transformerInfo.headDim)#N * queryLen * heads * headDim
        #jjj
        """#ccc this line is equal to == out = torch.einsum("nhqv,nvhd->nhqd", [attention, value]); out= out.permute(0, 2, 1, 3)
        
        we have 3 sets of words(value, key, query). now 2 of them (key and query) are consummed to create the attentionTable
        Each element of nqhd is obtained by multiplying nhqv,nvhd and then summing over the v dimension
        n and h exist in nhqv,nvhd and nqhd. so we can think qv,vd->qd
        so for a specific q and a specific d(in result tensor):for each v word in value, we multiply its "dth embedding value" to the "vth col of attentionTable" of row q, then sum them up
        qd=q1*1d+q2*2d+...
        for 1st embedding value of q word has weighed 1st embedding value of all other words due to their importance(calculated in attentionTable)
        """
        out = self.outLayer(out.reshape(N, queryLen, -1))
        out = out.reshape(N, queryLen, queryDim, self.transformerInfo.embedSize)
        # N * queryLen * embedSize

        return out
#%%

#%%
