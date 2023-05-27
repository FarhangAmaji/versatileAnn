# this is binaryClassifierToVariationalAutoencoderModule.py
#%%
import torch
import torch.nn as nn
import inspect

# Define a new module for binary classification
class binaryClassifierToVariationalAutoencoder(nn.Module):
    def __init__(self, vaeEncoder, hiddenSize):
        super(binaryClassifierToVariationalAutoencoder, self).__init__()
        self.inputArgs = [vaeEncoder, hiddenSize]
        self.vaeEncoder = vaeEncoder
        self.fc1 = nn.Linear(vaeEncoder.latentDim, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, 4*hiddenSize)
        self.fc3 = nn.Linear(4*hiddenSize, hiddenSize)
        self.fc4 = nn.Linear(hiddenSize, 1)
        self.lRelu = nn.LeakyReLU(negative_slope=0.05)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()

    def forward(self, x):
        mean, logvar, z = self.vaeEncoder(x)
        z = self.fc1(z)
        z = self.lRelu(z)
        z = self.dropout(z)
        z = self.fc2(z)
        z = self.lRelu(z)
        z = self.dropout(z)
        z = self.fc3(z)
        z = self.lRelu(z)
        z = self.dropout(z)
        z = self.fc4(z)
        z = self.sigmoid(z)
        return z

def trainBinaryClassifierToVariationalAutoencoder(model, trainInputs, trainOutputs, valInputs, valOutputs, binaryClassificationLoss,
                              modelOptimizer, numEpochs, batchSize, dropoutRate, device, patience=10, savePath=r'data\outputs\bestVaeModel'):
    model.dropout.p = dropoutRate
    bestValAccuracy = 0.0
    counter = 0
    
    for epoch in range(numEpochs):
        model.train()
        runningLoss = 0.0
    
        # Create random indexes for sampling
        indexes = torch.randperm(trainInputs.shape[0])
        for i in range(0, len(trainInputs), batchSize):
            modelOptimizer.zero_grad()
    
            # Create batch indexes
            batchIndexes = indexes[i:i + batchSize]
    
            batchInputs = trainInputs[batchIndexes].to(device)
            batchOutputs = trainOutputs[batchIndexes].to(device)
    
            # Forward pass through the model
            outputs = model(batchInputs)
    
            # Compute the classification loss
            loss = binaryClassificationLoss(outputs, batchOutputs)
    
            # Backward pass and optimization
            loss.backward()
            modelOptimizer.step()
    
            runningLoss += loss.item()
    
        epochLoss = runningLoss / len(trainInputs)
        print(f"Epoch [{epoch + 1}/{numEpochs}], aveItemLoss: {epochLoss:.6f}")
    
        # Evaluation
        model.eval()
        with torch.no_grad():
            valAccuracy = evaluateBinaryClassifierToVariationalAutoencoder(model, valInputs, valOutputs, batchSize, device)
    
            if valAccuracy > bestValAccuracy:
                bestValAccuracy = valAccuracy
                counter = 0
                torch.save({'className':model.__class__.__name__,'classDefinition':inspect.getsource(model.__class__),'inputArgs':model.inputArgs,'model':model.state_dict()}, savePath)
            elif valAccuracy < bestValAccuracy:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping! in {epoch + 1} epoch")
                    break
            else:
                pass
    
    print("Training finished.")

    # Load the best model model
    bestModel=torch.load(savePath)
    model.load_state_dict(bestModel['model'])
    
    return model

# Evaluate the binaryClassifierToVariationalAutoencoder on test data
def evaluateBinaryClassifierToVariationalAutoencoder(model, testInputs, testOutputs, batchSize, device):
    model.eval()
    correct = 0
    total = 0
    
    # Create random indexes for sampling
    indexes = torch.randperm(testInputs.shape[0])
    
    with torch.no_grad():
        for i in range(0, len(testInputs), batchSize):
            # Create batch indexes
            batchIndexes = indexes[i:i + batchSize]
    
            batchInputs = testInputs[batchIndexes].to(device)
            batchOutputs = testOutputs[batchIndexes].to(device)
    
            # Forward pass through the model
            outputs = model(batchInputs)
    
            # Convert probabilities to binary predictions (0 or 1)
            predictions = torch.round(outputs)
    
            # Count the number of correct predictions
            correct += (predictions == batchOutputs).sum().item()
            total += batchOutputs.size(0)
    
    accuracy = correct / total
    return accuracy