"""
Beispiel: Aktivitätsbasiertes Routing für ein neuronales Netzwerk
- Nur die aktivsten Blöcke eines Modells werden pro Input genutzt
- Demonstriert an einem einfachen Feedforward-Netz mit dynamischer Blockauswahl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        act = self.activation(out)
        # "Aktivität" als Mittelwert der Aktivierungen
        activity = act.abs().mean().item()
        return act, activity

class ActivityRoutedNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_blocks=4, top_k=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            DynamicBlock(in_features, hidden_features) for _ in range(num_blocks)
        ])
        self.top_k = top_k
        self.final = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # Berechne alle Block-Aktivierungen
        acts = []
        activities = []
        for block in self.blocks:
            act, activity = block(x)
            acts.append(act)
            activities.append(activity)
        # Wähle die top-k aktivsten Blöcke
        activities_tensor = torch.tensor(activities)
        topk_indices = torch.topk(activities_tensor, self.top_k).indices
        # Summiere die Ausgaben der aktivsten Blöcke
        combined = sum(acts[i] for i in topk_indices)
        out = self.final(combined / self.top_k)
        return out, activities_tensor, topk_indices

if __name__ == "__main__":
    torch.manual_seed(42)
    net = ActivityRoutedNet(in_features=8, hidden_features=16, out_features=2, num_blocks=4, top_k=2)
    x = torch.randn(1, 8)
    out, activities, topk = net(x)
    print(f"Aktivitäten pro Block: {activities}")
    print(f"Aktivste Blöcke: {topk}")
    print(f"Netz-Ausgabe: {out}")
