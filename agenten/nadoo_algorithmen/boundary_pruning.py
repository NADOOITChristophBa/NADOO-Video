import torch
import torch.nn as nn

class BoundaryPruner:
    """
    Applies boundary-based dynamic pruning on a model's parameters.
    Weights below threshold * max(abs(weights)) are zeroed.
    """
    def __init__(self, model):
        self.model = model

    def prune(self, threshold):
        for name, param in self.model.named_parameters():
            max_val = param.abs().max()
            if max_val == 0:
                continue
            mask = param.abs() > threshold * max_val
            param.data.mul_(mask.float())

    def __call__(self, x, threshold):
        self.prune(threshold)
        return self.model(x)
