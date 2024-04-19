import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class MlpMlcSm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MlpMlcSm, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # self.bn1 = nn.BatchNorm1d(hidden_size * 4)
        # Xavier initialization
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class MlpMlcMd(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MlpMlcMd, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc5 = nn.Linear(hidden_size // 2, output_size)

        # self.bn1 = nn.BatchNorm1d(hidden_size * 4)
        # Xavier initialization
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))  # Apply sigmoid activation
        return x

class MlpMlcLg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MlpMlcLg, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc5 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc8 = nn.Linear(hidden_size // 2, output_size)

        # self.bn1 = nn.BatchNorm1d(hidden_size * 4)
        # Xavier initialization
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6, self.fc7, self.fc8]:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = torch.sigmoid(self.fc8(x))  # Apply sigmoid activation
        return x