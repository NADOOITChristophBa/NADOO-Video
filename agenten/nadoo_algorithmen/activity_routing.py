import torch
import torch.nn as nn

class ActivityRoutedBlock(nn.Module):
    """
    Ein Block, der nur berechnet wird, wenn die Aktivität hoch genug ist.
    activity_fn: Funktion, die einen Aktivitätswert aus dem Input berechnet
    threshold: Schwelle, ab der der Block gerechnet wird
    """
    def __init__(self, in_features, out_features, activity_fn=None, threshold=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activity_fn = activity_fn or (lambda x: x.abs().mean().item())
        self.threshold = threshold
    def forward(self, x):
        activity = self.activity_fn(x)
        if activity > self.threshold:
            return self.linear(x), activity
        else:
            return x, activity

# Beispiel/Test
if __name__ == "__main__":
    block = ActivityRoutedBlock(4, 4, threshold=0.5)
    x = torch.tensor([[0.1, 0.2, 0.1, 0.2]])
    y, act = block(x)
    print(f"Aktivität: {act:.3f}, Output: {y}")
    x2 = torch.tensor([[10.0, -10.0, 10.0, -10.0]])
    y2, act2 = block(x2)
    print(f"Aktivität: {act2:.3f}, Output: {y2}")
