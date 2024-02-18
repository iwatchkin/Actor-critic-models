import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.__initialize_weights()
        self.linear_layer1 = nn.Linear(state_dim, 400)
        self.linear_layer2 = nn.Linear(400, 300)
        self.linear_layer3 = nn.Linear(300, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        hidden = self.linear_layer1(x)
        hidden = self.relu(hidden)
        hidden = self.linear_layer2(hidden)
        hidden = self.relu(hidden)
        hidden = self.linear_layer3(hidden)
        output = self.tanh(hidden)

        return output

    def __initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.linear_layer1 = nn.Linear(state_dim + action_dim, 400)
        self.linear_layer2 = nn.Linear(400, 300)
        self.linear_layer3 = nn.Linear(300, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.linear_layer1(x)
        hidden = self.relu(hidden)
        hidden = self.linear_layer2(hidden)
        hidden = self.relu(hidden)
        output = self.linear_layer3(hidden)

        return output


class ValueFunctionNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueFunctionNetwork, self).__init__()
        self.linear_layer1 = nn.Linear(state_dim, 400)
        self.linear_layer2 = nn.Linear(400, 300)
        self.linear_layer3 = nn.Linear(300, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.linear_layer1(x)
        hidden = self.relu(hidden)
        hidden = self.linear_layer2(hidden)
        hidden = self.relu(hidden)
        output = self.linear_layer3(hidden)

        return output