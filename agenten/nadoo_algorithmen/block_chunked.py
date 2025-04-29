import torch
import torch.nn as nn

class BlockChunkedRouting(nn.Module):
    """
    Teilt den Input in Chunks und berechnet nur die aktivsten Chunks pro Block.
    """
    def __init__(self, in_features, out_features, num_chunks=4, top_k=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_chunks = num_chunks
        self.top_k = top_k
        # compute per-chunk sizes for input and output
        q_in, r_in = divmod(in_features, num_chunks)
        self.in_chunk_sizes = [q_in+1]*r_in + [q_in]*(num_chunks-r_in)
        q_out, r_out = divmod(out_features, num_chunks)
        self.out_chunk_sizes = [q_out+1]*r_out + [q_out]*(num_chunks-r_out)
        self.blocks = nn.ModuleList([
            nn.Linear(self.in_chunk_sizes[i], self.out_chunk_sizes[i])
            for i in range(num_chunks)
        ])
    def forward(self, x):
        # x: [batch, in_features]
        x_chunks = torch.split(x, self.in_chunk_sizes, dim=1)
        activities = [chunk.abs().mean().item() for chunk in x_chunks]
        topk_indices = torch.topk(torch.tensor(activities), self.top_k).indices
        out_chunks = []
        for i, chunk in enumerate(x_chunks):
            if i in topk_indices:
                out_chunks.append(self.blocks[i](chunk))
            else:
                # zero tensor with correct output chunk size
                batch_size = x.size(0)
                out_chunks.append(torch.zeros(batch_size, self.out_chunk_sizes[i],
                                              device=x.device, dtype=x.dtype))
        return torch.cat(out_chunks, dim=1), activities

# Beispiel/Test
if __name__ == "__main__":
    model = BlockChunkedRouting(8, 8, num_chunks=4, top_k=2)
    x = torch.randn(1, 8)
    out, acts = model(x)
    print("Input:", x)
    print("Aktivitäten:", acts)
    print("Output:", out)
