import torch.nn as nn

class ModelFactory(nn.Module):
    def __init__(self, model: nn.Module, num_classes=1000, **kargs):
        super(ModelFactory, self).__init__()
        self.model = model
        self.linear = nn.Linear(1000, num_classes)

    def forward(self, x):
        assert x.shape[1] == 3, "Only three-channel data is supported."
        out = self.model(x)
        out = self.linear(out)
        return out