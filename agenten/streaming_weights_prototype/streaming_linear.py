import torch
import torch.nn as nn
import os
import numpy as np

class StreamingLinear(nn.Module):
    """
    Linear-Layer, der die Gewichtsmatrix blockweise von der Festplatte lädt.
    Nur ein Block ist gleichzeitig im Speicher (Demo-Zweck).
    """
    def __init__(self, in_features, out_features, chunk_size=1024, weight_file=None, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chunk_size = chunk_size
        self.weight_file = weight_file
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_param', None)
        # Anzahl der Blöcke
        self.num_chunks = (out_features + chunk_size - 1) // chunk_size

    def forward(self, input):
        outputs = []
        for chunk_idx in range(self.num_chunks):
            start = chunk_idx * self.chunk_size
            end = min((chunk_idx + 1) * self.chunk_size, self.out_features)
            weight_chunk = self._load_weight_chunk(start, end)
            # input: [batch, in_features], weight_chunk: [chunk, in_features]
            out_chunk = torch.matmul(input, weight_chunk.t())
            if self.bias:
                out_chunk += self.bias_param[start:end]
            outputs.append(out_chunk)
        return torch.cat(outputs, dim=1)

    def _load_weight_chunk(self, start, end):
        # Annahme: Die Gewichte liegen als float32 und Zeilen-major in einer Binärdatei
        assert self.weight_file is not None, "weight_file muss gesetzt sein!"
        size = (end - start, self.in_features)
        offset = start * self.in_features * 4  # float32 = 4 bytes
        with open(self.weight_file, 'rb') as f:
            f.seek(offset)
            chunk = np.frombuffer(f.read((end - start) * self.in_features * 4), dtype=np.float32)
            chunk = chunk.reshape(size)
        return torch.from_numpy(chunk)
