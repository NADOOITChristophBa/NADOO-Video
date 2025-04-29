import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nadoo_algorithmen.async_distributed import AsyncDistributedInference
from nadoo_algorithmen.activity_routing import ActivityRoutedBlock
from nadoo_algorithmen.block_chunked import BlockChunkedRouting
from nadoo_algorithmen.dynamic_quantization import DynamicQuantizedLinear

class SmartComboModel(nn.Module):
    """
    Kombiniert alle fortschrittlichen Algorithmen:
    - Erst Block-Chunked Routing (nur aktive Chunks)
    - Dann dynamische Quantisierung je nach Aktivität
    - Dann aktivitätsbasiertes Block-Skipping
    """
    def __init__(self, in_features=8, hidden=8, out_features=4, num_chunks=4, top_k=2):
        super().__init__()
        self.block_chunked = BlockChunkedRouting(in_features, hidden, num_chunks=num_chunks, top_k=top_k)
        self.quant = DynamicQuantizedLinear(hidden, hidden)
        self.activity_block = ActivityRoutedBlock(hidden, out_features, threshold=0.2)
    def forward(self, x):
        # Block-Chunked Routing
        x, activities = self.block_chunked(x)
        # Dynamische Quantisierung (verwende Mittelwert der Aktivitäten)
        mean_activity = sum(activities) / len(activities)
        x = self.quant(x, mean_activity)
        # Aktivitätsbasiertes Block-Skipping
        out, act = self.activity_block(x)
        return out, dict(chunk_activities=activities, quant_activity=mean_activity, block_activity=act)

if __name__ == "__main__":
    # Beispiel: Asynchrone verteilte Inferenz mit dem Kombi-Modell
    model = SmartComboModel()
    devices = [torch.device('cpu')]
    async_inf = AsyncDistributedInference(model, devices)
    # Simuliere Batch von Inputs
    batch = [torch.randn(1, 8) for _ in range(5)]
    results = async_inf.infer(batch)
    for i, (out, infos) in enumerate(results):
        print(f"Sample {i}: Output = {out}, Infos = {infos}")
