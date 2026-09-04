MECHANISM: Additive embedding gauge fixing

HYPOTHESIS: Anchoring the position-zero embedding will reduce the model from 1,596 to 1,588 parameters while retaining at least 99% accuracy, because a shared shift between token and positional embeddings leaves hidden states unchanged and changes tied-head logits only by a class-independent constant.

INTENDED_EDIT: Replace the positional embedding table with a table whose zero-position vector is fixed at zero, and initialize the remaining embeddings with a gauge-equivalent transformation.

EVIDENCE: The 1,596-parameter model achieved 99.99%, while deleting the capacity-bearing final LayerNorm bias failed at 93.36%; this motivates removing an exact eight-parameter embedding redundancy instead of another affine ablation.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
class AnchoredPositionEmbedding(nn.Module):
    """Position embeddings with the additive token/position gauge fixed at position zero."""

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.rest = nn.Embedding(max_seq_len - 1, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        zero = self.rest.weight.new_zeros((1, self.rest.embedding_dim))
        table = torch.cat((zero, self.rest.weight), dim=0)
        return F.embedding(positions, table)


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = AnchoredPositionEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
=======
        self.apply(self._init_weights)

        # Gauge-transform a standard initialization: E' = E + P[0],
        # P'[i] = P[i] - P[0]. Inputs are unchanged, and tied-head logits
        # gain only the same scalar for every vocabulary item.
        with torch.no_grad():
            anchor = torch.empty(cfg.d_model, device=self.token_emb.weight.device)
            nn.init.normal_(anchor, mean=0.0, std=0.02)
            self.token_emb.weight.add_(anchor)
            self.pos_emb.rest.weight.sub_(anchor)
>>>>>>> REPLACE