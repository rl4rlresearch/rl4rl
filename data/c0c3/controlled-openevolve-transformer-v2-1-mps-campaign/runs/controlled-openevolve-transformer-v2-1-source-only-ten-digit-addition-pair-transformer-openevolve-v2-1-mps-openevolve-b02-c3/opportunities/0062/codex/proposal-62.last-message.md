MECHANISM: Full per-head Q/K diagonal gauge fixing

HYPOTHESIS: A 827-parameter transformer will retain at least 99% accuracy because the verified 829-parameter design reached 99.88%, and fixing the remaining key-channel scale in each head removes two reciprocal Q/K gauge degrees without reducing attention expressivity.

INTENDED_EDIT: Reconstruct the verified six-dimensional, rank-five lexical design and fix all three key-channel scale gauges per head instead of two.

EVIDENCE: The 829-parameter six-dimensional design achieved 99.88%, while reducing its terminal scales failed; extending the already-qualified Q/K gauge fixing targets exact attention-score reparameterizations while preserving the load-bearing five-dimensional token representation, seven-unit MLP, and three terminal scales.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with two reciprocal Q/K scales fixed per head."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        out_features = 3 * d_model
        head_dim = d_model // n_head
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(2, head_dim))
        }
=======
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with every reciprocal Q/K scale fixed per head."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        out_features = 3 * d_model
        head_dim = d_model // n_head
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(head_dim)
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 3
        )
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-model", type=int, default=8)
=======
    p.add_argument("--d-model", type=int, default=6)
>>>>>>> REPLACE