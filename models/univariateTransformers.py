#%% imports
# models\multivariateTransformers.py
import sys
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(baseFolder)
sys.path.append(parent_folder)
from versatileAnn import ann
from versatileAnn.layers import linLReluNormDropout, linLSigmoidNormDropout
import torch
from torch import nn
#%% define model
"""
originial github: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py
some parts of code been corrected also has been modified and to suit multivariateTransformers
"""
class multiHeadAttention(nn.Module):#kkk make it suitable for multivariate
    def __init__(self, embedSize, heads):#kkk correct explanations for this
        super(multiHeadAttention, self).__init__()
        '''#ccc its important to know the network is independent of sequence length because we didnt make any layer(linear)
        which takes input equal to sequence length
        '''
        self.embedSize = embedSize
        self.heads = heads
        self.head_dim = embedSize // heads

        assert (self.head_dim * heads == embedSize), "Embedding size needs to be divisible by heads number"

        self.valueLayer = nn.Linear(embedSize, embedSize)
        self.keyLayer = nn.Linear(embedSize, embedSize)
        self.queryLayer = nn.Linear(embedSize, embedSize)
        self.outLayer = nn.Linear(embedSize, embedSize)

    def forward(self, query, key, value, mask):
        'multiHeadAttention'
        'this can be used for encoder and both decoder attentions'
        '#ccc in 2nd attention in decoder, embedded output is query'
        # Get batchSize
        N = query.shape[0]

        valueLen, queryLen = value.shape[1], query.shape[1]
        """#ccc the valueLen and keyLen in both attentions in decoder are the same
        but the queryLen in 1st attention in decoder is same as them and in 2nd attention not the same"""
        value = self.valueLayer(value)  # N * valueLen * embedSize
        key = self.keyLayer(key)  # N * valueLen * embedSize
        query = self.queryLayer(query)  # N * queryLen * embedSize

        # Split the embedding into self.heads different pieces
        value = value.reshape(N, valueLen, self.heads, self.head_dim)
        key = key.reshape(N, valueLen, self.heads, self.head_dim)
        query = query.reshape(N, queryLen, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
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
        if mask is not None:#kkk dont send make if its not needed
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embedSize ** (1 / 2)), dim=3)
        # attention shape: N * heads * queryLen * valueLen
        '''#ccc we Normalize energy value for better stability
        note after this we would have attention[0,0,0,:].sum()==1
        
        note one question comes to mind "we are doing softmax, what difference would deviding by
        'embedSize ** (1 / 2)' make,finally we are doing softmax":? yes because softmax does averaging exponentially so smaller
        value get so less softmax output
        '''

        out = torch.einsum("nhqv,nvhd->nqhd", [attention, value]).reshape(N, queryLen, self.heads * self.head_dim)#N * queryLen * heads * head_dim
        """#ccc this line is equal to == out = torch.einsum("nhqv,nvhd->nhqd", [attention, value]); out= out.permute(0, 2, 1, 3)
        
        we have 3 sets of words(value, key, query). now 2 of them (key and query) are consummed to create the attentionTable
        Each element of nqhd is obtained by multiplying nhqv,nvhd and then summing over the v dimension
        n and h exist in nhqv,nvhd and nqhd. so we can think qv,vd->qd
        so for a specific q and a specific d(in result tensor):for each v word in value, we multiply its "dth embedding value" to the "vth col of attentionTable" of row q, then sum them up
        qd=q1*1d+q2*2d+...
        for 1st embedding value of q word has weighed 1st embedding value of all other words due to their importance(calculated in attentionTable)
        """
        out = self.outLayer(out)
        # N * queryLen * embedSize

        return out
