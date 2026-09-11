MECHANISM: Six-coordinate tied lexical bottleneck

HYPOTHESIS: Constraining the tied token embedding and classifier to six learned coordinates will produce a 1,164-parameter transformer with at least 99% accuracy, because lexical representation need not occupy the full seven-dimensional residual quotient while the successful seven-dimensional attention and MLP computation remains intact.

INTENDED_EDIT: Replace the shared assumption that tokens require every zero-mean residual direction with a six-dimensional learned token/output code, and reduce the aligned final classifier bias accordingly.

EVIDENCE: The 1,325-parameter fixed-token-basis design retained 99.97% accuracy after removing token-subspace tilt, and the current 1,279-parameter model reaches 99.97%; meanwhile, direct attention operators fell to 89.02%, motivating a substantial reduction at the large tied lexical interface while preserving the qualified attention mechanism.

<<<<<<< SEARCH
class FactorizedTokenEmbedding(nn.Module):
    """Gauge-fixed low-rank token map shared with the output classifier."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank != embedding_dim - 1:
            raise ValueError("rank must equal embedding_dim - 1")
=======
class FactorizedTokenEmbedding(nn.Module):
    """Fixed-subspace low-rank token map shared with the output classifier."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank < 1 or rank >= embedding_dim:
            raise ValueError("rank must be between one and embedding_dim - 1")
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 2
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
        self.final_bias = nn.Parameter(torch.zeros(cfg.d_model - 1))
=======
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim)
        )
>>>>>>> REPLACE