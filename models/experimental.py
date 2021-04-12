import torch
import torch.nn as nn


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None



def attempt_load(weights, map_location=None):
    model = Ensemble()
    ckpt = torch.load(weights, map_location=map_location)

    model.append(ckpt['model'].float().eval())
    return model[-1]
