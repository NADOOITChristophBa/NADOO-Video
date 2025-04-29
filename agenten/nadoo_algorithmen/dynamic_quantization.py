import torch
import torch.nn as nn

class DynamicQuantizedLinear(nn.Module):
    """
    Linear-Layer, der die Quantisierung je nach Aktivität dynamisch anpasst.
    """
    def __init__(self, in_features, out_features, bits_fn=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bits_fn = bits_fn or (lambda activity: 8 if activity > 0.5 else 1)
    def quantize(self, w, bits):
        if bits == 32:
            return w
        elif bits == 8:
            scale = 255 / (w.max() - w.min() + 1e-8)
            return ((w - w.min()) * scale).round() / scale + w.min()
        elif bits == 1:
            return (w > 0).float() * w.max()
        else:
            raise ValueError('Nur 32, 8 oder 1 Bit unterstützt (Demo)')
    def forward(self, x, activity):
        bits = self.bits_fn(activity)
        wq = self.quantize(self.linear.weight, bits)
        return torch.matmul(x, wq.t()) + self.linear.bias

# Beispiel/Test
if __name__ == "__main__":
    model = DynamicQuantizedLinear(4, 4)
    x = torch.randn(1, 4)
    activity = x.abs().mean().item()
    out = model(x, activity)
    print(f"Aktivität: {activity:.3f}, Output: {out}")
