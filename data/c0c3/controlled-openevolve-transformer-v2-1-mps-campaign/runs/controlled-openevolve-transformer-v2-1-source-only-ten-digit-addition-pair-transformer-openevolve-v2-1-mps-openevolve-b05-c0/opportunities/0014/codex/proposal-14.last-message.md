MECHANISM: Quotient-aware positional-embedding gauge fixing

HYPOTHESIS: Removing one LayerNorm-invariant coordinate from the final positional vector will reduce the model from 1634 to 1633 parameters while retaining at least 99% accuracy when virtual eight-coordinate AdamW preserves the full embedding’s initialization and optimizer dynamics.

INTENDED_EDIT: Represent the final positional vector with seven learned coordinate differences and one fixed zero, initialize it as the exact gauge-equivalent of an eight-coordinate draw, and optimize it with the existing gauge-aware AdamW and gradient clipping.

EVIDENCE: Ordinary positional gauge fixing collapsed to 17.26%, but the analogous output-bias gauge recovered from 73.43% to 99.85% when virtual full-coordinate AdamW was introduced; this directly motivates applying the successful optimizer treatment to the exact positional gauge.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class GaugeFixedPositionalEmbedding(nn.Module):
    """Embedding with the final row represented modulo common feature shifts."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings - 1, embedding_dim)
        )
        self.last_row = nn.Parameter(torch.empty(embedding_dim - 1))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        full = torch.empty(
            self.num_embeddings,
            self.embedding_dim,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        nn.init.normal_(full, mean=0.0, std=0.02)
        self.weight.copy_(full[:-1])
        self.last_row.copy_(full[-1, :-1] - full[-1, -1])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        final = F.pad(self.last_row, (0, 1)).unsqueeze(0)
        return F.embedding(idx, torch.cat((self.weight, final), dim=0))


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = GaugeFixedPositionalEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedPositionalEmbedding):
            module.reset_parameters()
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [block.attn.proj_bias for block in model.blocks]
=======
    gauge_parameters = [block.attn.proj_bias for block in model.blocks]
    gauge_parameters.append(model.pos_emb.last_row)
>>>>>>> REPLACE