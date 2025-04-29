"""
Beispiel: Kombiniertes blockweises Streaming und aktivitätsbasiertes Routing
- Das Modell besteht aus mehreren Blöcken (z.B. Layer), jeder Block ist in Chunks (z.B. Channel-, Patch-, Head-Gruppen) unterteilt
- Pro Block werden nur die aktivsten Chunks geladen und berechnet
- Die Blöcke werden nacheinander (seriell/streaming) verarbeitet, sodass nie das ganze Modell gleichzeitig im Speicher ist
"""
import torch
import torch.nn as nn

class ChunkedDynamicBlock(nn.Module):
    def __init__(self, in_features, out_features, num_chunks=4, top_k_chunks=2):
        super().__init__()
        self.num_chunks = num_chunks
        self.top_k_chunks = top_k_chunks
        # Jeder Chunk ist ein Teil der Gewichtsmatrix
        self.chunks = nn.ModuleList([
            nn.Linear(in_features // num_chunks, out_features // num_chunks)
            for _ in range(num_chunks)
        ])

    def forward(self, x):
        # Split input in Chunks (z.B. Channel-wise)
        x_chunks = torch.chunk(x, self.num_chunks, dim=1)
        activities = []
        # 1. Grobe Voraktivierung (z.B. Norm des Inputs pro Chunk)
        for chunk in x_chunks:
            approx_activity = chunk.abs().mean().item()
            activities.append(approx_activity)
        # 2. Wähle die aktivsten Chunks
        activities_tensor = torch.tensor(activities)
        topk_indices = torch.topk(activities_tensor, self.top_k_chunks).indices
        # 3. Berechne nur für diese Chunks das Forward
        activations = [self.chunks[i](x_chunks[i]) for i in topk_indices]
        # 4. Kombiniere die Outputs (hier: einfach konkatenieren)
        output = torch.cat(activations, dim=1)
        return output, activities_tensor, topk_indices

class BlockChunkedActivityRoutedNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_blocks=3, num_chunks=4, top_k_chunks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            ChunkedDynamicBlock(
                in_features if i == 0 else hidden_features,
                hidden_features,
                num_chunks=num_chunks,
                top_k_chunks=top_k_chunks
            ) for i in range(num_blocks)
        ])
        self.final = nn.Linear(hidden_features // num_chunks * top_k_chunks, out_features)

    def forward(self, x):
        for block in self.blocks:
            x, activities, topk = block(x)
            print(f"Block Aktivitäten: {activities}, Top-K Chunks: {topk}")
        out = self.final(x)
        return out

if __name__ == "__main__":
    torch.manual_seed(42)
    net = BlockChunkedActivityRoutedNet(8, 8, 2, num_blocks=3, num_chunks=4, top_k_chunks=2)
    x = torch.randn(1, 8)
    out = net(x)
    print(f"Netz-Ausgabe: {out}")
