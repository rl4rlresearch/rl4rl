MECHANISM: Five-coordinate tied lexical bottleneck

HYPOTHESIS: Reducing the qualified six-coordinate token embedding and classifier to five learned coordinates will produce a 1,049-parameter transformer with at least 99% accuracy, because the six-coordinate design achieved 99.92% while preserving ample continuous capacity to distinguish the vocabulary.

INTENDED_EDIT: Apply the qualified five-direction pre-MLP bias configuration, reduce the tied token/output code from six coordinates to five, and align the final learned bias with that lexical subspace.

EVIDENCE: The 1,164-parameter six-coordinate design achieved 99.92% accuracy, while the seven-coordinate design achieved 99.97%; the negligible accuracy change motivates testing the next lexical rank before narrowing attention, whose direct-operator reduction failed at 89.02%.

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
class QuotientBiasLayerNorm(nn.Module):
    """Scale-fixed LayerNorm retaining six zero-mean bias directions."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 3:
            raise ValueError("normalized_shape must be at least three")

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 2))

        basis = torch.zeros(normalized_shape, normalized_shape - 2)
        for col in range(normalized_shape - 2):
=======
class QuotientBiasLayerNorm(nn.Module):
    """Scale-fixed LayerNorm retaining five zero-mean bias directions."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 4:
            raise ValueError("normalized_shape must be at least four")

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 3))

        basis = torch.zeros(normalized_shape, normalized_shape - 3)
        for col in range(normalized_shape - 3):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 3
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