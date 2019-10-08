import torch.nn as nn


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size=103):

        super(MultiLayerPerceptron, self).__init__()

        ######
        hidden_size = 64
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.act2 = nn.Sigmoid()

        ######

    def forward(self, features):

        ######

        layer1 = self.fc1(features)
        layer1 = self.act1(layer1)
        layer2 = self.fc2(layer1)
        return self.act2(layer2)

        ######
