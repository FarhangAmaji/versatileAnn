# ---- imports
# models\multivariateTransformers.py
import sys
import os
parentFolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentFolder)
from versatileAnn import ann
import torch
from torch import nn
# ---- define model
#kkk rename the classes
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
    
    def makeTrgMask(self, outputLen):
        trgMask = torch.tril(torch.ones((outputLen, outputLen))).unsqueeze(1).unsqueeze(-1)
        return trgMask.to(self.device)
    
    def positionalEmbedding2d(self, maxRows, maxCols):#kkk add shapes and comment
        positionRows = (torch.arange(maxRows).reshape(maxRows, 1).repeat(1, maxCols))
        positionEnc = torch.zeros(maxRows, maxCols, self.embedSize)
        # Calculate positional encodings for rows and columns
        for d in range(0, self.embedSize, 2):
            denominator = torch.pow(10000.0, torch.tensor(d, dtype=torch.float32) / self.embedSize)
            positionEnc[:, :, d] = torch.sin(positionRows / denominator)
            positionEnc[:, :, d + 1] = torch.cos(positionRows / denominator)
        return positionEnc.to(self.device)
    
class multiHeadAttention(nn.Module):
    def __init__(self, transformerInfo, queryDim, valueDim):#kkk correct comments
        super(multiHeadAttention, self).__init__()
        '''#ccc its important to know the network is independent of sequence length because we didnt make any layer(linear)
        which takes input equal to sequence length
        '''
        self.transformerInfo= transformerInfo
        self.valueLayer = nn.Linear(valueDim * transformerInfo.embedSize, valueDim * transformerInfo.embedSize)
        self.keyLayer = nn.Linear(valueDim * transformerInfo.embedSize, valueDim * transformerInfo.embedSize)
        self.queryLayer = nn.Linear(queryDim * transformerInfo.embedSize, queryDim * transformerInfo.embedSize)
        self.outLayer = nn.Linear(queryDim * transformerInfo.embedSize, queryDim * transformerInfo.embedSize)

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
        value = value.reshape(N, valueLen, valueDim, self.transformerInfo.heads, self.transformerInfo.headDim)
        key = key.reshape(N, valueLen, valueDim, self.transformerInfo.heads, self.transformerInfo.headDim)
        query = query.reshape(N, queryLen, queryDim, self.transformerInfo.heads, self.transformerInfo.headDim)

        energy = torch.einsum("nqxhd,nkyhd->nhqxky", [query, key])
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
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        energy=energy.reshape(N, self.transformerInfo.heads, queryLen, queryDim, -1)
        attention = torch.softmax(energy / (self.transformerInfo.embedSize ** (1 / 2)), dim=4)
        attention=attention.reshape(N, self.transformerInfo.heads, queryLen, queryDim, valueLen, valueDim)
        # attention shape: N * heads * queryLen * valueLen
        '''#ccc we Normalize energy value for better stability
        note after this we would have attention[0,0,0,:].sum()==1
        
        note one question comes to mind "we are doing softmax, what difference would deviding by
        'embedSize ** (1 / 2)' make,finally we are doing softmax":? yes because softmax does averaging exponentially so smaller
        value get so less softmax output
        '''
        out = torch.einsum("nhqxvy,nvyhd->nqxhd", [attention, value]).reshape(N, queryLen, queryDim, self.transformerInfo.heads * self.transformerInfo.headDim)#N * queryLen * heads * headDim
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


class layerNorm2D(nn.Module):
    def __init__(self, dim1, dim2):
        super(layerNorm2D, self).__init__()
        self.layerNorms = nn.ModuleList([nn.LayerNorm(dim2) for _ in range(dim1)])
        self.dim1= dim1
    
    def forward(self, x):
        normalized=torch.zeros_like(x)
        for i in range(self.dim1):
            normalized[:, :, i] = self.layerNorms[i](x[:, :, i])
        return normalized

class transformerBlock(nn.Module):
    def __init__(self, transformerInfo, queryDim, valueDim, dropout):
        '#ccc this is used in both encoder and decoder'
        super(transformerBlock, self).__init__()
        self.transformerInfo= transformerInfo
        self.attention = multiHeadAttention(transformerInfo, queryDim, valueDim)
        self.norm1 = layerNorm2D(queryDim, transformerInfo.embedSize)
        self.norm2 = layerNorm2D(queryDim, transformerInfo.embedSize)

        self.feedForward = nn.Sequential(nn.Linear(queryDim * transformerInfo.embedSize , transformerInfo.forwardExpansion * queryDim *  transformerInfo.embedSize),
            nn.LeakyReLU(negative_slope=.05),
            nn.Linear(transformerInfo.forwardExpansion * queryDim *  transformerInfo.embedSize, queryDim * transformerInfo.embedSize))

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        'transformerBlock'
        attention = self.attention(query, key, value, mask)
        N, seqLen, dim, _ = query.shape

        # Add skip connection, run through normalization and finally dropout
        x = self.norm1(attention + query)
        '#ccc the query is used in skip connection'
        forward = self.feedForward(x.reshape(N, seqLen, -1)).reshape(N, seqLen, dim, -1)
        out = self.dropout(self.norm2(forward + x))
        return out
    

class Encoder(nn.Module):
    def __init__(self, transformerInfo):
        super(Encoder, self).__init__()
        self.transformerInfo= transformerInfo
        self.embeddings = nn.Linear(transformerInfo.inputDim, transformerInfo.inputDim * transformerInfo.embedSize)
        self.layers = nn.ModuleList(
            [transformerBlock(transformerInfo, queryDim=transformerInfo.inputDim, valueDim=transformerInfo.inputDim, dropout=transformerInfo.dropoutRate+i*(1-transformerInfo.dropoutRate)/transformerInfo.encoderLayersNum) for i in range(transformerInfo.encoderLayersNum)])

    def forward(self, x):
        'Encoder'
        N, inputSeqLength, inputDim = x.shape
        x = self.embeddings(x).reshape(N, inputSeqLength, inputDim, self.transformerInfo.embedSize) + self.transformerInfo.encoderPositionalEmbedding
            
        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            x = layer(x, x, x, None)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, transformerInfo, dropout):
        super(DecoderBlock, self).__init__()
        self.transformerInfo= transformerInfo
        self.norm = layerNorm2D(transformerInfo.outputDim, transformerInfo.embedSize)
        self.attention = multiHeadAttention(transformerInfo, transformerInfo.outputDim, transformerInfo.outputDim)
        self.transformerBlock = transformerBlock(transformerInfo, queryDim=transformerInfo.outputDim, valueDim=transformerInfo.inputDim,dropout=dropout)
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
        self.embeddings = nn.Linear(transformerInfo.outputDim, transformerInfo.outputDim * transformerInfo.embedSize)

        self.layers = nn.ModuleList(
            [DecoderBlock(transformerInfo, dropout=transformerInfo.dropoutRate+i*(.9-transformerInfo.dropoutRate)/transformerInfo.decoderLayersNum) for i in range(transformerInfo.decoderLayersNum)])
        self.outLayer = nn.Linear(transformerInfo.outputDim * transformerInfo.embedSize, transformerInfo.outputDim)

    def forward(self, outputSeq, outputOfEncoder):
        'Decoder'
        N, outputSeqLength, outputDim = outputSeq.shape
        outputSeq = self.embeddings(outputSeq).reshape(N, outputSeqLength, outputDim, self.transformerInfo.embedSize) + self.transformerInfo.decoderPositionalEmbedding[:outputSeqLength]

        for layer in self.layers:
            outputSeq = layer(outputSeq, outputOfEncoder, outputOfEncoder, self.transformerInfo.makeTrgMask(outputSeqLength))
        '#ccc note outputSeq is query, outputOfEncoder is value and key'

        outputSeq = self.outLayer(outputSeq.reshape(N,outputSeqLength,-1))
        return outputSeq

class multivariateTransformer(ann):
    def __init__(self, transformerInfo):
        super(multivariateTransformer, self).__init__()

        self.transformerInfo= transformerInfo
        self.backcastLen=transformerInfo.inpLen
        self.forecastLen=transformerInfo.outputLen
        self.timeSeriesMode=True
        self.transformerMode=True
        self.encoder = Encoder(transformerInfo)
        self.decoder = Decoder(transformerInfo)

    def forward(self, src, trg):
        'Transformer'
        assert src.shape[2]==self.transformerInfo.inputDim,f'src dim={src.shape[2]} and transformerInfo inputDim ={self.transformerInfo.inputDim}; u should either change ur inputs or create suitable transformerInfo'
        assert trg.shape[2]==self.transformerInfo.outputDim,f'trg dim={trg.shape[2]} and transformerInfo outputDim ={self.transformerInfo.outputDim}; u should either change ur targets or create suitable transformerInfo'
        src=src.to(self.device)
        trg=trg.to(self.device)
        encSrc = self.encoder(src)
        out = self.decoder(trg, encSrc)
        return out
    
    def forwardForUnknown(self, src, outputLen):
        self.eval()
        output=torch.zeros(1, 1, self.transformerInfo.outputDim).to(self.device)
        for i in range(outputLen):
            newOutput=self.forward(src, output)#kkk is this correct
            output=torch.cat([output,newOutput[:,-1].unsqueeze(0)],dim=1)#kkk find a pair of input and output with known and pass it to the trained model: doesnt reproduce the same results
        return output#kkk if I could get same results; also correct forwardForUnknown for univariate
    
    def forwardForUnknownStraight(self, src, outputLen):
        self.eval()
        src=src.to(self.device)
        N, _, _ = src.shape
        encInps=self.encoder(src)
        outputs=torch.zeros(N,outputLen,self.transformerInfo.outputDim).to(self.device)
        return self.decoder(outputs,encInps)

# ----

# ----
