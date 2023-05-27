# this is variationalAutoencoderModule.py
#%% imports
import torch
import torch.nn as nn
import inspect
#%% Define variational Autoencoder (VAE) architecture
class variationalEncoder(nn.Module):
    def __init__(self, inputSize, latentDim):
        super(variationalEncoder, self).__init__()
        self.inputArgs = [inputSize, latentDim]
        self.latentDim = latentDim
        self.fc1 = nn.Linear(inputSize, 4*inputSize)
        self.fc2 = nn.Linear(4*inputSize, inputSize)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.fcMean = nn.Linear(inputSize, latentDim)
        self.fcLogvar = nn.Linear(inputSize, latentDim)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        mean = self.fcMean(x)
        logvar = self.fcLogvar(x)
        z = self.reparameterize(mean, logvar)
        return mean, logvar, z

class variationalDecoder(nn.Module):
    def __init__(self, latentDim, inputSize):
        super(variationalDecoder, self).__init__()
        self.inputArgs = [latentDim, inputSize]
        self.fc1 = nn.Linear(latentDim, 4*inputSize)
        self.fc2 = nn.Linear(4*inputSize, inputSize)
        self.fc3 = nn.Linear(inputSize, inputSize)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()

    def forward(self, z):
        z = self.fc1(z)
        z = self.sigmoid(z)
        z = self.dropout(z)
        z = self.fc2(z)
        z = self.sigmoid(z)
        z = self.dropout(z)
        z = self.fc3(z)
        return z

class variationalAutoencoder(nn.Module):
    def __init__(self, inputSize, latentDim):
        super(variationalAutoencoder, self).__init__()
        self.inputArgs = [inputSize, latentDim]
        self.encoder = variationalEncoder(inputSize, latentDim)
        self.decoder = variationalDecoder(latentDim, inputSize)

    def forward(self, x):
        mean, logvar, z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, mean, logvar

# Define KL divergence loss
def klDivergenceLoss(mean, logvar):
    klLoss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return klLoss

def trainVae(model, trainInputs, trainOutputs, valInputs, valOutputs, criterion, optimizer, numEpochs, batchSize, dropoutRate, device, patience, savePath):
    model.train()
    bestValLoss = float('inf')
    counter = 0
    
    model.encoder.dropout.p = dropoutRate
    model.decoder.dropout.p = dropoutRate
    
    for epoch in range(numEpochs):
        trainLoss = 0.0
        
        # Create random indexes for sampling
        indexes = torch.randperm(trainInputs.shape[0])

        for i in range(0, trainInputs.shape[0], batchSize):
            # Create batch indexes
            batchIndexes = indexes[i:i+batchSize]

            # Extract batch inputs and outputs
            batchInputs = trainInputs[batchIndexes].to(device)
            appliedBatchSize, inputSize = batchInputs.shape

            # Forward pass
            reconstructed, mean, logvar = model(batchInputs)

            # Compute losses
            reconLoss = criterion(reconstructed, batchInputs)
            klLoss = klDivergenceLoss(mean, logvar)
            totalLoss = reconLoss + klLoss

            # Backward pass and optimization
            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()

            trainLoss += totalLoss.item()

        # Compute average loss for the epoch
        trainLoss /= trainInputs.shape[0]

        # Validation loop
        valLoss = evaluateVae(model, valInputs, valOutputs, criterion, batchSize, device)

        # Print training and validation loss for the epoch
        print(f'Epoch [{epoch+1}/{numEpochs}], Train Loss: {trainLoss:.4f}, Val Loss: {valLoss:.4f}')
        
        if valLoss < bestValLoss:
            bestValLoss = valLoss
            counter = 0
            torch.save({'className':model.__class__.__name__,'classDefinition':inspect.getsource(model.__class__),'inputArgs':model.inputArgs,'model':model.state_dict()}, savePath)
        elif valLoss > bestValLoss:
            counter += 1
            if counter >= patience:
                print(f"Early stopping! in {epoch+1} epoch")
                break
        else:
            pass
    
    print("Training finished.")
    
    # Load the best model
    bestModel=torch.load(savePath)
    model.load_state_dict(bestModel['model'])
    
    # Return the best model
    return model

def evaluateVae(model, inputs, outputs, criterion, batchSize, device):
    model.eval()
    testLoss = 0.0

    with torch.no_grad():
        for i in range(0, inputs.shape[0], batchSize):
            # Extract batch inputs and outputs
            batchInputs = inputs[i:i+batchSize].to(device)

            # Forward pass
            reconstructed, mean, logvar = model(batchInputs)

            # Compute losses
            reconLoss = criterion(reconstructed, batchInputs)
            klLoss = klDivergenceLoss(mean, logvar)
            totalLoss = reconLoss + klLoss

            testLoss += totalLoss.item()

        # Compute average loss on test data
        testLoss /= inputs.shape[0]

    print(f'Test Loss: {testLoss:.4f}')
    return testLoss