MECHANISM: Gauge-fix one LayerNorm-invariant positional scalar

HYPOTHESIS: Removing one redundant scalar from the final positional embedding will reduce the model from 1636 to 1635 parameters while retaining at least 99% accuracy, because shifting every coordinate of one positional vector equally is canceled by every pre-normalization and the final LayerNorm.

INTENDED_EDIT: Replace the positional embedding with an equivalent 1635-parameter embedding that fixes one coordinate of its final row while preserving the successful model’s initialization distribution and all computational paths.

EVIDENCE: The 1636-parameter design achieved 99.84%, while reducing MLP capacity reached only 95.44% and attention/normalization changes fell further; this motivates removing a single exact positional gauge degree of freedom without altering those sensitive components.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
class GaugeFixedPositionalEmbedding(nn.Module):
    """Embedding with one unobservable row-wise offset fixed to zero."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if num_embeddings < 2 or embedding_dim < 2:
            raise ValueError("gauge-fixed embedding requires at least two rows and columns")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.prefix = nn.Parameter(torch.empty(num_embeddings - 1, embedding_dim))
        self.last_coords = nn.Parameter(torch.empty(embedding_dim - 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            full = torch.empty(
                self.num_embeddings,
                self.embedding_dim,
                device=self.prefix.device,
                dtype=self.prefix.dtype,
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            self.prefix.copy_(full[:-1])
            self.last_coords.copy_(full[-1, :-1] - full[-1, -1])

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        last_row = torch.cat((self.last_coords, self.last_coords.new_zeros(1)))
        weight = torch.cat((self.prefix, last_row.unsqueeze(0)), dim=0)
        return F.embedding(indices, weight)


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = GaugeFixedPositionalEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedPositionalEmbedding):
            module.reset_parameters()
>>>>>>> REPLACE