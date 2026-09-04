MECHANISM: Incremental pre-MLP bias gauge fixing

HYPOTHESIS: The qualified 1,044-parameter model will retain at least 99% accuracy with 1,043 parameters when the pre-MLP LayerNorm bias is reduced from five directions to four, because the omitted constant contribution is representable by `fc1`’s learned bias.

INTENDED_EDIT: Apply the qualified five-coordinate lexical bottleneck and affine-free pre-attention LayerNorm, then remove one learned pre-MLP bias coefficient.

EVIDENCE: The five-coordinate model achieved 99.93%, its affine-free pre-attention variant achieved 99.90% with 1,044 parameters, and earlier incremental pre-MLP bias reductions from seven to six and six to five retained 99.95% and 99.97%, respectively.

<<<<<<< SEARCH
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
=======
class QuotientBiasLayerNorm(nn.Module):
    """Scale-fixed LayerNorm retaining four zero-mean bias directions."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 5:
            raise ValueError("normalized_shape must be at least five")

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 4))

        basis = torch.zeros(normalized_shape, normalized_shape - 4)
        for col in range(normalized_shape - 4):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = GaugeFixedLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 2
        )
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 3
        )
>>>>>>> REPLACE