import torch.nn as nn
#%% model
class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
            )
        
    def forward(self, x):
        return self.net(x)

#%% model1
class PredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(81, 256),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )
        
    def forward(self, x):
        return self.net(x)
#%% model2
class PredictionModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(80, 256),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )
        
    def forward(self, x):
        return self.net(x)
    
#%% model3
class PredictionModel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(800, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )
        
    def forward(self, x):
        return self.net(x)

#%% model4
class PredictionModel4(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(400, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )
        
    def forward(self, x):
        return self.net(x)
    

#%% model5 for gridsearch
class PredictionModel5(nn.Module):
    def __init__(self, neuronsize):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(800, 128*neuronsize),
            nn.ReLU(),
            nn.Linear(128*neuronsize, 64*neuronsize),
            nn.ReLU(),
            nn.Linear(64*neuronsize, 32*neuronsize),
            nn.ReLU(),
            nn.Linear(32*neuronsize, 16*neuronsize),
            nn.ReLU(),
            nn.Linear(16*neuronsize, 8*neuronsize),
            nn.ReLU(),
            nn.Linear(8*neuronsize, 1)
            )
        
    def forward(self, x):
        return self.net(x)
    

#%% model2
class PredictionModel_single_SDS(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )
        
    def forward(self, x):
        return self.net(x)
    
#%% model6 model for particular wavelength
class PredictionModel6(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )
        
    def forward(self, x):
        return self.net(x)