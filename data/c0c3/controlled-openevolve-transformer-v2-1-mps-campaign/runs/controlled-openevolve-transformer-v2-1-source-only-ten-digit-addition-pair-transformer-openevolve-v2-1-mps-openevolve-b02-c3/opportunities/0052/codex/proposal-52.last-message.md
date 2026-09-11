MECHANISM: Eight-feature nonlinear carry bottleneck

HYPOTHESIS: Reducing the qualified nine-unit GELU MLP to eight units will produce a 987-parameter transformer with at least 99% accuracy, because the nine-unit model achieved 99.94% accuracy and this adjacent reduction preserves its two specialized attention heads and five-dimensional token representation.

INTENDED_EDIT: Reconstruct the qualified four-key-fix and two-coordinate final-bias reductions, then reduce `d_ff` from nine to eight, removing 15 additional MLP parameters.

EVIDENCE: The 1,002-parameter design with `d_ff=9` achieved 99.94% accuracy, while the preceding `d_ff=10` design achieved 99.96%; the adjacent width reduction is the narrowest informative continuation of the successful MLP compression path.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with one reciprocal Q/K scale fixed per head."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        out_features = 3 * d_model
        head_dim = d_model // n_head
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(
            (d_model + head * head_dim) * in_features
            for head in range(n_head)
        )
=======
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

        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(
            (d_model + channel) * in_features
            for channel in sorted(selected_key_channels)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim)
        )
=======
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 2)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        x = x + self.final_bias @ self.token_emb.basis.transpose(0, 1)
        logits = self.token_emb.logits(x)
=======
        x = self.ln_f(x)
        final_bias = F.pad(self.final_bias, (0, 2))
        x = x + final_bias @ self.token_emb.basis.transpose(0, 1)
        logits = self.token_emb.logits(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=11)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE