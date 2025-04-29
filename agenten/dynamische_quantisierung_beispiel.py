"""
Beispiel: Dynamische, adaptive Quantisierung pro Chunk
Chunk-Gewichte werden je nach Aktivität unterschiedlich quantisiert (simuliert)
"""
import torch
import torch.nn as nn
import numpy as np

def quantize_tensor(tensor, bits):
    if bits == 32:
        return tensor
    elif bits == 8:
        # Simuliere 8-bit Quantisierung
        scale = 255 / (tensor.max() - tensor.min() + 1e-8)
        return ((tensor - tensor.min()) * scale).round() / scale + tensor.min()
    elif bits == 1:
        # Simuliere Binary Quantisierung
        return (tensor > 0).float() * tensor.max()
    else:
        raise ValueError('Nur 32, 8 oder 1 Bit unterstützt (Demo)')

class QuantizedChunk(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    def forward(self, x, bits):
        wq = quantize_tensor(self.weight, bits)
        return torch.matmul(x, wq.t()) + self.bias

if __name__ == "__main__":
    torch.manual_seed(42)
    chunk = QuantizedChunk(8, 4)
    x = torch.randn(1, 8)
    for bits in [32, 8, 1]:
        out = chunk(x, bits)
        print(f"Output mit {bits}-bit Quantisierung: {out}")
