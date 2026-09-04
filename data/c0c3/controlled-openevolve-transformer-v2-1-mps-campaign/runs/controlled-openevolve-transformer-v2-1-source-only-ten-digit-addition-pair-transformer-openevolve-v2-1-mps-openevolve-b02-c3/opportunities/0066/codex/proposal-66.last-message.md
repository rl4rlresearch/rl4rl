MECHANISM: Intra-head value/output scale gauge fixing

HYPOTHESIS: A 827-parameter transformer will retain at least 99% accuracy because fixing a second value-channel coefficient within the already-qualified first attention head removes another reciprocal V/output-projection scale degree without constraining attention routing.

INTENDED_EDIT: Extend the successful 828-parameter value-path gauge fixing to the first two value channels of the first head, while leaving the second head unconstrained.

EVIDENCE: Fixing one value coefficient in the first head achieved 99.97%, whereas applying the corresponding constraint to the second head collapsed to 5.21%; this motivates testing whether gauge-fixing tolerance is head-specific by keeping both reductions within the successful head.

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

        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(
            (d_model + channel) * in_features
            for channel in sorted(selected_key_channels)
        )
=======
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K and two first-head V scales fixed."""

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
        selected_rows = {
            d_model + channel for channel in selected_key_channels
        }
        selected_rows.update(
            2 * d_model + offset for offset in range(min(2, head_dim))
        )

        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(
            row * in_features for row in sorted(selected_rows)
        )
>>>>>>> REPLACE