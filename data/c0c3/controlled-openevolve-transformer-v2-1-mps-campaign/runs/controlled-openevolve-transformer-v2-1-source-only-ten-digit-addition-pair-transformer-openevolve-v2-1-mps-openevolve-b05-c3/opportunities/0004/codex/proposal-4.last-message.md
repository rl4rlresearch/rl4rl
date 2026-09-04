MECHANISM: Gauge-fix one functionally redundant positional-embedding scalar

HYPOTHESIS: Removing one common-mode degree of freedom from the final positional embedding will reduce parameters from 1644 to 1643 while retaining at least 99% accuracy, because every LayerNorm is invariant to per-token shifts shared across all eight residual channels.

INTENDED_EDIT: Replace the positional embedding with an otherwise identical learned embedding whose final row fixes its last coordinate to zero, while initializing that row to a functionally equivalent gauge transformation of the baseline initialization.

EVIDENCE: The d_ff=12 baseline reached 99.96% accuracy, whereas reducing nonlinear capacity or changing attention biases failed; this motivates preserving the proven architecture and function class while removing only a LayerNorm-invisible positional degree of freedom.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
class LastRowGaugedEmbedding(nn.Module):
    """Embedding with one LayerNorm-invisible positional degree fixed to zero."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings - 1, embedding_dim))
        self.last_weight = nn.Parameter(torch.empty(embedding_dim - 1))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        last = F.pad(self.last_weight, (0, 1)).unsqueeze(0)
        weight = torch.cat((self.weight, last), dim=0)
        return F.embedding(idx, weight)


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = LastRowGaugedEmbedding(cfg.max_seq_len, cfg.d_model)
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
        elif isinstance(module, LastRowGaugedEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(module.num_embeddings, module.embedding_dim)
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(full[:-1])
                module.last_weight.copy_(full[-1, :-1] - full[-1, -1])
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--run-name", type=str, default="repro_l1_d8_ff12")
    p.add_argument("--run-dir", type=Path, default=Path("results/runs/repro_l1_d8_ff12"))
=======
    p.add_argument("--run-name", type=str, default="repro_l1_d8_ff12_posgauge1")
    p.add_argument("--run-dir", type=Path, default=Path("results/runs/repro_l1_d8_ff12_posgauge1"))
>>>>>>> REPLACE