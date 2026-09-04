MECHANISM: Locally initialized tied-token common-mode gauge

HYPOTHESIS: Fixing one tied-token embedding scalar at zero while independently initializing every remaining scalar will reduce the model from 1,631 to 1,630 parameters and retain at least 99% accuracy by avoiding the global initialization correlation introduced by the previous gauge-preserving subtraction.

INTENDED_EDIT: Store the tied token embedding with one fewer scalar, reconstruct only its final scalar as zero, initialize the compact learned coordinates normally, and use the reconstructed matrix through a parameterless tied output projection.

EVIDENCE: The globally shifted token-embedding anchor reached only 84.68%, while a localized positional common-mode anchor reached 99.93%; this motivates retaining the exact token-embedding gauge reduction but localizing its initialization effect to one scalar.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = PositionAnchoredEmbedding(cfg.max_seq_len, cfg.d_model)
=======
class TokenAnchoredEmbedding(nn.Embedding):
    """Tied token embedding with one global common-mode scalar anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self.weight.detach().flatten()[:-1].clone())

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 1)).view(
            self.num_embeddings, self.embedding_dim
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class TiedTokenLinear(nn.Module):
    """Parameterless output projection using the reconstructed token matrix."""

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weight)


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = TokenAnchoredEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = PositionAnchoredEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        # Weight tying with the reconstructed input embedding matrix.
        self.lm_head = TiedTokenLinear()
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = self.lm_head(x, self.token_emb.full_weight())
>>>>>>> REPLACE