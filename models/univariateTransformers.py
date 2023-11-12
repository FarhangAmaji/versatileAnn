# ---- imports
# models\univariateTransformers.py
import sys
import os
parentFolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentFolder)
from versatileAnn import ann
import torch
from torch import nn
# ---- define model
"""
originial github: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py
some parts of code been corrected also has been modified and to suit multivariateTransformers
"""
class TransformerInfo:
    def __init__(self, embedSize=32, heads=8, forwardExpansion=4, encoderLayersNum=6, decoderLayersNum=6, dropoutRate=.6, inpLen=10, outputLen=10):
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedSize = embedSize
        self.heads = heads
        self.forwardExpansion = forwardExpansion
        self.dropoutRate = dropoutRate
        self.headDim = embedSize // heads
        assert (self.headDim * heads == embedSize), "Embedding size needs to be divisible by heads number"

        self.inpLen= inpLen
        self.outputLen= outputLen
        self.encoderLayersNum= encoderLayersNum
        self.decoderLayersNum= decoderLayersNum
        self.encoderPositionalEmbedding=self.positionalEmbedding1d(inpLen)
        self.decoderPositionalEmbedding=self.positionalEmbedding1d(outputLen+1)#+1 is for last data from input added to the first of output
        '#ccc for regression data we dont have startToken like in nlp so we assume that the last input is the first output; therefore we add +1 to outputLen'
    
    def makeTrgMask(self, outputLen):
        trgMask = torch.tril(torch.ones((outputLen, outputLen)))
        return trgMask.to(self.device)
    
    def positionalEmbedding1d(self, maxRows):
        even_i = torch.arange(0, self.embedSize, 2).float()# shape: embedSize//2
        denominator = torch.pow(10000, even_i/self.embedSize)
        position = (torch.arange(maxRows).reshape(maxRows, 1))
                          
        even_PE = torch.sin(position / denominator)# shape: maxLen * embedSize//2; each row is for a position
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)# shape: maxLen * embedSize//2 * 2
        PE = torch.flatten(stacked, start_dim=1, end_dim=2).to(self.device)# shape: maxLen * embedSize
        """#ccc
        PE for position i: [sin(i\denominator[0]),cos(i\denominator[0]), sin(i\denominator[1]),cos(i\denominator[1]), sin(i\denominator[2]),cos(i\denominator[2]),...]        
        thus PE for even position i starts at sin(i\denominator[0]) and goes to 0 because the denominator[self.embedSize-1] is a really large number sin(0)=0
        thus PE for odd position i starts at cos(i\denominator[0]) and goes to 1 because the denominator[self.embedSize-1] is a really large number cos(0)=1
        note for different'i's the sin(i\denominator[0]) just circulates and its not confined to sth like (i/maxRows*2*pi)
        therefore the maxRows plays really doesnt affect the positional encoding and for each position we can get PE for position i without having maxRows
        #kkk why the positional embeddings for a specific position i, each embedding element different?
        """
        return PE #ccc in order PE to be summable with other embeddings, we fill for the rest of values upto maxLen with some padding

class multiHeadAttention(nn.Module):
    def __init__(self, transformerInfo):
        super(multiHeadAttention, self).__init__()
        '''#ccc its important to know the network is independent of sequence length because we didnt make any layer(linear)
        which takes input equal to sequence length
        '''
        self.transformerInfo= transformerInfo
        self.valueLayer = nn.Linear(transformerInfo.embedSize, transformerInfo.embedSize)
        self.keyLayer = nn.Linear(transformerInfo.embedSize, transformerInfo.embedSize)
        self.queryLayer = nn.Linear(transformerInfo.embedSize, transformerInfo.embedSize)
        self.outLayer = nn.Linear(transformerInfo.embedSize, transformerInfo.embedSize)

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

        # Split the embedding into heads different pieces
        value = value.reshape(N, valueLen, self.transformerInfo.heads, self.transformerInfo.headDim)
        key = key.reshape(N, valueLen, self.transformerInfo.heads, self.transformerInfo.headDim)
        query = query.reshape(N, queryLen, self.transformerInfo.heads, self.transformerInfo.headDim)

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
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.transformerInfo.embedSize ** (1 / 2)), dim=3)
        # attention shape: N * heads * queryLen * valueLen
        '''#ccc we Normalize energy value for better stability
        note after this we would have attention[0,0,0,:].sum()==1
        
        note one question comes to mind "we are doing softmax, what difference would deviding by
        'embedSize ** (1 / 2)' make,finally we are doing softmax":? yes because softmax does averaging exponentially so smaller
        value get so less softmax output
        '''

        out = torch.einsum("nhqv,nvhd->nqhd", [attention, value]).reshape(N, queryLen, self.transformerInfo.heads * self.transformerInfo.headDim)#N * queryLen * heads * headDim
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

class TransformerBlock(nn.Module):
    def __init__(self, transformerInfo, dropout):
        '#ccc this is used in both encoder and decoder'
        super(TransformerBlock, self).__init__()
        self.transformerInfo= transformerInfo
        self.attention = multiHeadAttention(transformerInfo)
        self.norm1 = nn.LayerNorm(transformerInfo.embedSize)
        self.norm2 = nn.LayerNorm(transformerInfo.embedSize)

        self.feedForward = nn.Sequential(nn.Linear(transformerInfo.embedSize, transformerInfo.forwardExpansion * transformerInfo.embedSize),
            nn.LeakyReLU(negative_slope=.05),
            nn.Linear(transformerInfo.forwardExpansion * transformerInfo.embedSize, transformerInfo.embedSize))

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        'TransformerBlock'
        attention = self.attention(query, key, value, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.norm1(attention + query)
        '#ccc the query is used in skip connection'
        forward = self.feedForward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    

class Encoder(nn.Module):
    def __init__(self, transformerInfo):
        super(Encoder, self).__init__()
        self.transformerInfo= transformerInfo
        self.embeddings = nn.Linear(1, transformerInfo.embedSize)
        self.layers = nn.ModuleList(
            [TransformerBlock(transformerInfo, dropout=transformerInfo.dropoutRate+i*(1-transformerInfo.dropoutRate)/transformerInfo.encoderLayersNum) for i in range(transformerInfo.encoderLayersNum)])

    def forward(self, x):
        'Encoder'
        N, inputSeqLength = x.shape
        x = self.embeddings(x.unsqueeze(2)) + self.transformerInfo.encoderPositionalEmbedding
            
        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            x = layer(x, x, x, None)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, transformerInfo, dropout):
        super(DecoderBlock, self).__init__()
        self.transformerInfo= transformerInfo
        self.norm = nn.LayerNorm(transformerInfo.embedSize)
        self.attention = multiHeadAttention(transformerInfo)
        self.transformerBlock = TransformerBlock(transformerInfo,dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, trgMask):
        'DecoderBlock'
        attention = self.attention(query, query, query, trgMask)
        query = self.dropout(self.norm(attention + query))
        out = self.transformerBlock(query, key, value, None)
        """#ccc note here we pass the mask equal to None the same mask as encoder because the attention table is 
        outSeqLen*inpSeqLen so no need to prevent future words
        
        if it wasnt regression which we dont have empty spaces in our sequence like nlp then we would have assigned
        negative infinity to the paddings"""
        return out

class Decoder(nn.Module):
    def __init__(self, transformerInfo):
        super(Decoder, self).__init__()
        self.transformerInfo= transformerInfo
        self.embeddings = nn.Linear(1, transformerInfo.embedSize)

        self.layers = nn.ModuleList(
            [DecoderBlock(transformerInfo, dropout=transformerInfo.dropoutRate+i*(.9-transformerInfo.dropoutRate)/transformerInfo.decoderLayersNum) for i in range(transformerInfo.decoderLayersNum)])
        self.outLayer = nn.Linear(transformerInfo.embedSize, 1)

    def forward(self, outputSeq, outputOfEncoder):
        'Decoder'
        outputSeqLength=outputSeq.shape[1]
        outputSeq = self.embeddings(outputSeq.unsqueeze(2)) + self.transformerInfo.decoderPositionalEmbedding[:outputSeqLength]

        for layer in self.layers:
            outputSeq = layer(outputSeq, outputOfEncoder, outputOfEncoder, self.transformerInfo.makeTrgMask(outputSeqLength))
        '#ccc note outputSeq is query, outputOfEncoder is value and key'

        outputSeq = self.outLayer(outputSeq)
        return outputSeq

class univariateTransformer(ann):
    def __init__(self, transformerInfo):
        super(univariateTransformer, self).__init__()

        self.transformerInfo= transformerInfo
        self.backcastLen=transformerInfo.inpLen
        self.forecastLen=transformerInfo.outputLen
        self.timeSeriesMode=True
        self.transformerMode=True
        self.encoder = Encoder(transformerInfo)
        self.decoder = Decoder(transformerInfo)

    def forward(self, src, trg):
        'Transformer'
        src=src.to(self.device)
        trg=trg.to(self.device)
        encSrc = self.encoder(src)
        out = self.decoder(trg, encSrc)
        return out.squeeze(2)
    
    def forwardForUnknown(self, src, outputLen):
        self.eval()
        if len(src.shape)==1:
            src=src.unsqueeze(0)
        output=src[:,-1].unsqueeze(0).to(self.device)
        for i in range(outputLen):
            newOutput=self.forward(src, output)
            output=torch.cat([output,newOutput[:,-1].unsqueeze(0)],dim=1)
        return output
    
    def forwardForUnknownStraight(self, src, outputLen):
        self.eval()
        if len(src.shape)==1:
            src=src.unsqueeze(0)
        src=src.to(self.device)
        outputs=src[:,-1].unsqueeze(0).to(self.device)
        outputs=torch.cat([outputs,torch.zeros(1,outputLen).to(self.device)],dim=1)
        encInps=self.encoder(src)
        return self.decoder(outputs,encInps)
# ----

# ----
