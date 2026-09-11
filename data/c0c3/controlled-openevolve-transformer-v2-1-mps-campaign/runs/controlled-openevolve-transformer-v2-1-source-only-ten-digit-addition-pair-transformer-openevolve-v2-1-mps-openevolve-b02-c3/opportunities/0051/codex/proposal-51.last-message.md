MECHANISM: Nine-feature nonlinear carry bottleneck

HYPOTHESIS: Reducing the qualified 10-unit GELU MLP to nine units will produce a 1,002-parameter transformer with at least 99% accuracy, because the 10-unit model achieved 99.96% accuracy and retained substantial margin above the threshold.

INTENDED_EDIT: Reconstruct the qualified four-key-fix, two-coordinate final-bias reduction, then reduce `d_ff` from 10 to 9, removing 15 additional MLP parameters.

EVIDENCE: The qualified 1,017-parameter model with `d_ff=10` achieved 99.96% accuracy; testing the adjacent width of nine is the narrowest informative continuation of the most successful large parameter reduction.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with three reciprocal Q/K scales fixed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        out_features = 3 * d_model
        head_dim = d_model // n_head
        selected_key_channels = {
            head * head_dim for head in range(n_head)
        }
        selected_key_channels.add(1)
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
    p.add_argument("--d-ff", type=int, default=9)
>>>>>>> REPLACE