# dropoutNetModule.py
import torch
import torch.nn as nn

class dropoutNet(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(dropoutNet, self).__init__()

        self.fc1 = nn.Linear(inputSize, 4*inputSize)
        self.lRelu = nn.LeakyReLU(negative_slope=0.05)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(4*inputSize, 4*inputSize)
        self.fc3 = nn.Linear(4*inputSize, outputSize)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.lRelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.lRelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def trainModel(model, trainInputs, trainOutputs, valInputs, valOutputs, criterion, optimizer, numEpochs, batchSize, numSamples, dropoutRate, device, patience, savePath):
    model.train()
    bestValAccuracy = 0.0
    counter = 0
    
    model.dropout.p = dropoutRate
    for epoch in range(numEpochs):
        runningLoss = 0.0
        
        # Create random indexes for sampling
        indexes = torch.randperm(trainInputs.shape[0])
        for i in range(0, len(trainInputs), batchSize):
            optimizer.zero_grad()
            # Create batch indexes
            batchIndexes = indexes[i:i+batchSize]
            
            batchTrainInputs = trainInputs[batchIndexes].to(device)
            batchTrainOutputs = trainOutputs[batchIndexes].to(device)
            
            batchTrainOutputsPred = model.forward(batchTrainInputs)
            loss = criterion(batchTrainOutputsPred, batchTrainOutputs)
            
            loss.backward()
            optimizer.step()
            
            runningLoss += loss.item()
        
        epochLoss = runningLoss / (len(trainInputs) / batchSize)
        print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {epochLoss:.4f}")
        
        with torch.no_grad():
            valAccuracy = evaluateModel(model, valInputs, valOutputs, numSamples, batchSize, device, dropoutRate)
            
            if valAccuracy > bestValAccuracy:
                bestValAccuracy = valAccuracy
                counter = 0
                torch.save(model.state_dict(), savePath)
            elif valAccuracy < bestValAccuracy:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping! in {epoch+1} epoch")
                    break
            else:
                pass
    
    print("Training finished.")
    
    # Load the best model
    model.load_state_dict(torch.load(savePath))
    
    # Return the best model
    return model

def evaluateModel(model, inputs, outputs, numSamples, batchSize, device, dropoutRate):
    model.eval()
    model.dropout.p = dropoutRate
    
    with torch.no_grad():
        correct = 0
        
        for i in range(0, len(inputs), batchSize):
            batchInputs = inputs[i:i+batchSize].to(device)
            batchOutputs = outputs[i:i+batchSize].to(device)
            appliedBatchSize, outputSize = batchOutputs.shape
            
            batchOutputsPred = torch.zeros((numSamples, appliedBatchSize)).to(device)
            
            batchOutputsPred = torch.stack(tuple(map(lambda x: model.forward(x).squeeze(), [batchInputs] * numSamples)))
            
            meanOutput = batchOutputsPred.mean(dim=0)
            predictions = torch.reshape((meanOutput >= 0.5).float(), (-1, outputSize))
            
            correct += (predictions == batchOutputs).sum().item()
        
        accuracy = correct / len(inputs)
        print(f"Accuracy: {accuracy:.4f}")
        
        return accuracy
