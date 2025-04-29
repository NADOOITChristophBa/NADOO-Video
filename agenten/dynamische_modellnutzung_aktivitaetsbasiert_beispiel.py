"""
Beispiel: Dynamische, aktivitätsbasierte Modellnutzung
Nur aktive Blöcke werden berechnet, inaktive werden übersprungen.
"""
import torch
import torch.nn as nn

class DynamicBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
    def forward(self, x):
        out = self.linear(x)
        act = self.activation(out)
        activity = act.abs().mean().item()
        return act, activity

class DynamicNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_blocks=4, threshold=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([
            DynamicBlock(in_features, hidden_features) for _ in range(num_blocks)
        ])
        self.threshold = threshold
        self.final = nn.Linear(hidden_features, out_features)
    def forward(self, x):
        for i, block in enumerate(self.blocks):
            act, activity = block(x)
            print(f"Block {i}: Aktivität = {activity:.3f}")
            if activity > self.threshold:
                x = act
            else:
                print(f"Block {i} übersprungen!")
        out = self.final(x)
        return out

if __name__ == "__main__":
    torch.manual_seed(42)
    net = DynamicNet(8, 16, 2, num_blocks=4, threshold=0.2)
    x = torch.randn(1, 8)
    out = net(x)
    print(f"Netz-Ausgabe: {out}")
