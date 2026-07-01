import torch
import torch.nn as nn

class FeedForwardANN(nn.Module):
    def __init__(self, input_size=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GaussianMF(nn.Module):
    def __init__(self, in_features, n_mfs):
        super().__init__()
        self.mu = nn.Parameter(torch.rand(in_features, n_mfs))
        self.sigma = nn.Parameter(torch.rand(in_features, n_mfs))

    def forward(self, x):
        x = x.unsqueeze(2) 
        gauss = torch.exp(-((x - self.mu)**2) / (2 * self.sigma**2))
        return gauss.view(x.size(0), -1) 

class ANFISThenANN(nn.Module):
    def __init__(self, in_features=8, n_mfs=6, hidden_size=32):
        super().__init__()
        self.fuzzy = GaussianMF(in_features, n_mfs)
        self.ann = nn.Sequential(
            nn.Linear(in_features * n_mfs, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        fuzzy_output = self.fuzzy(x)
        return self.ann(fuzzy_output)
