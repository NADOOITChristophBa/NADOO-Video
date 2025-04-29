import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from diffusers_helper.streaming_linear import StreamingLinear

# Hilfsfunktion: Erstelle eine große Gewichtsmatrix und speichere sie blockweise ab

def create_weight_file(weight_file, out_features, in_features):
    weights = np.random.randn(out_features, in_features).astype(np.float32)
    with open(weight_file, 'wb') as f:
        f.write(weights.tobytes())
    return weights

if __name__ == "__main__":
    in_features = 512
    out_features = 4096  # Viel größer als typische mobile VRAM
    chunk_size = 1024
    weight_file = "test_weights.bin"

    # Erstelle Gewichtsmatrix
    weights = create_weight_file(weight_file, out_features, in_features)

    # Input Batch
    x = torch.randn(2, in_features)

    # Modell
    model = StreamingLinear(in_features, out_features, chunk_size=chunk_size, weight_file=weight_file, bias=True)
    out = model(x)
    print("Output shape:", out.shape)
    # Vergleich mit Standard-Linear
    ref = torch.nn.Linear(in_features, out_features, bias=True)
    ref.weight.data = torch.from_numpy(weights)
    ref.bias.data = model.bias_param.data
    out_ref = ref(x)
    print("Max. Abweichung zum Referenzmodell:", (out - out_ref).abs().max().item())
