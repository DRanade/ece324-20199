# Assignment 2 skeleton code
# This code shows you how to use the 'argparse' library to read in parameters

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Command Line Arguments
parser = argparse.ArgumentParser(description='generate training and validation data for assignment 2')
parser.add_argument('trainingfile', help='name stub for training data and label output in csv format', default="train")
parser.add_argument('validationfile', help='name stub for validation data and label output in csv format', default="valid")
parser.add_argument('numtrain', help='number of training samples',type= int,default=200)
parser.add_argument('numvalid', help='number of validation samples',type= int,default=20)
parser.add_argument('-seed', help='random seed', type= int,default=10)
parser.add_argument('-learningrate', help='learning rate', type= float,default=0.1)
parser.add_argument('-actfunction', help='activation functions', choices=['sigmoid', 'relu', 'linear'], default='linear')
parser.add_argument('-numepoch', help='number of epochs', type= int,default=50)

args = parser.parse_args()

traindataname = args.trainingfile + "data.csv"
trainlabelname = args.trainingfile + "label.csv"

print("training data file name: ", traindataname)
print("training label file name: ", trainlabelname)

validdataname = args.validationfile + "data.csv"
validlabelname = args.validationfile + "label.csv"

print("validation data file name: ", validdataname)
print("validation label file name: ", validlabelname)

print("number of training samples = ", args.numtrain)
print("number of validation samples = ", args.numvalid)

print("learning rate = ", args.learningrate)
print("number of epoch = ", args.numepoch)

print("activation function is ", args.actfunction)

tdata = np.loadtxt(traindataname, delimiter=',')[:args.numtrain]
tlabels = np.loadtxt(trainlabelname, delimiter=',')[:args.numtrain]
vdata = np.loadtxt(validdataname, delimiter=',')[:args.numvalid]
vlabels = np.loadtxt(validlabelname, delimiter=',')[:args.numvalid]
train_acc = np.zeros(args.numepoch)
train_loss = np.zeros(args.numepoch)
val_acc = np.zeros(args.numepoch)
val_loss = np.zeros(args.numepoch)
np.random.seed(args.seed)


class SNC(nn.Module):
    def __init__(self):
        super(SNC, self).__init__()
        self.fc1 = nn.Linear(9, 1)
        return

    def forward(self, I):
        return self.fc1(I)


# copied accuracy function because my history arrays are now obsolete since PyTorch tensors will store info
def accuracy(predictions, label):
    total_corr = 0
    index = 0
    for c in predictions.flatten():
        if (c.item() > 0.5):
            r = 1
        else:
            r = 0
        if (r == label[index].item()):
            total_corr += 1
        index += 1
    return total_corr / len(label)


Tdata = torch.from_numpy(tdata)
print("Shape of Tdata = ", Tdata.size())
Tlabels = torch.from_numpy(tlabels)
print("Shape of Tlabels = ", Tlabels.size())
Vdata = torch.from_numpy(vdata)
print("Shape of Vdata = ", Vdata.size())
Vlabels = torch.from_numpy(vlabels)
print("Shape of Vlabels = ", Vlabels.size())

neuron = SNC()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(neuron.parameters(), lr=args.learningrate)

lossRec = []
vlossRec = []
nRec = []
trainAccRec = []
validAccRec = []

# removed my activation function function since we are only doing a Linear activation function with nn.Linear

# removed my partial function (which calculated gradient) because the optimizer is now doing that

for i in range(args.numepoch):
    # set up optimizer with blank gradients
    optimizer.zero_grad()
    # predict with the current weights
    predict = neuron(Tdata.float())
    # evaluate the loss for the predictions
    loss = loss_function(input=predict.squeeze(), target=Tlabels.float())
    # calculate the gradients per the loss function we chose
    loss.backward()
    # changes the weights one step in the right direction
    optimizer.step()

    # Evaluate performance section

    # check training accuracy
    trainAcc = accuracy(predict, Tlabels)
    # check validation accuracy
    predict = neuron(Vdata.float())
    vloss = loss_function(input=predict.squeeze(), target=Vlabels.float())
    validAcc = accuracy(predict, Vlabels)

    # store these accuracy values for future plots
    lossRec.append(loss)
    vlossRec.append(vloss)
    nRec.append(i)
    trainAccRec.append(trainAcc)
    validAccRec.append(validAcc)

    # deleted my own forward functions an0d gradient calculations
    # deleted my own stats calculations (i.e. loss, accuracy for both training and validation sets)

# Show the final weights and bias for fun
print("Neuron weights: ", neuron.fc1.weight)
print("Neuron bias: ", neuron.fc1.bias)

# Plot losses
plt.figure()
plt.plot(nRec, lossRec, label='Training Loss')
plt.plot(nRec, vlossRec, label='Validation Loss')
plt.title("Losses as Function of Epoch")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()

# Plot accuracies
plt.figure()
plt.plot(nRec, trainAccRec, label='Training Accuracy')
plt.plot(nRec, validAccRec, label='Validation Accuracy')
plt.ylim(0, 1)
plt.title("Accuracy as Function of Epoch")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()
