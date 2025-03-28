import torch.nn as nn

class DQN(nn.Module):
    def __init__(
        self,
        num_stops: int = 100,
        embedded_dim: int = 32, 
        hidden_dim: int = 64,
        output_dim: int = 100
    ):
        super(DQN, self).__init__()

        self.emb = nn.Embedding(num_stops, embedded_dim)
        self.layer1 = nn.Linear(embedded_dim, hidden_dim, bias = True)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim, bias = True)
        self.layer3 = nn.Linear(hidden_dim, output_dim, bias = True)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.emb(x)
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        x = self.activation(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        
        return x
