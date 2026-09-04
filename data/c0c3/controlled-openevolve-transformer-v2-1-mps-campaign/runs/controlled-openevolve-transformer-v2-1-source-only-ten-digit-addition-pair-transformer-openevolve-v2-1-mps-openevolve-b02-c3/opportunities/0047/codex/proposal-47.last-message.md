MECHANISM: Single-head maximum-distance bias pruning

HYPOTHESIS: The qualified four-key-fix architecture will retain at least 99% accuracy with 1,033 parameters when one head’s maximum-distance relative-bias coefficient is fixed at zero, because that coefficient affects only one attention logit at the full context length.

INTENDED_EDIT: Apply the qualified two-key-coefficient-per-head gauge fixing, then store all relative-bias coefficients except the second head’s farthest-distance coefficient and reconstruct that coefficient as a fixed zero.

EVIDENCE: The four-key-fix design achieved 99.65% accuracy with 1,034 parameters, while two different 1,033-parameter reductions targeting Q/K or query-bias gauges collapsed; this motivates testing a distinct, narrowly supported routing parameter.

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
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
=======
        self.relative_bias_width = max_seq_len - 1
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_width - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias[:, : seqlen - 1],
            ),
            dim=1,
        )
=======
        relative_bias_coeff = torch.cat(
            (self.relative_bias, self.relative_bias.new_zeros(1))
        ).view(self.n_head, self.relative_bias_width)
        relative_bias = torch.cat(
            (
                relative_bias_coeff.new_zeros(self.n_head, 1),
                relative_bias_coeff[:, : seqlen - 1],
            ),
            dim=1,
        )
>>>>>>> REPLACE