MECHANISM: Fifth reciprocal Q/K channel-scale gauge fix

HYPOTHESIS: Fixing a third key-channel coefficient in the first attention head will reduce the model from 1,034 to 1,033 parameters while retaining at least 99% accuracy, because that channel’s learned query projection and query bias preserve the reciprocal Q/K scaling freedom.

INTENDED_EDIT: Extend the qualified symmetric four-coefficient gauge fixing by fixing one additional nonzero key-projection coefficient in the first attention head.

EVIDENCE: The current model achieved 99.65% accuracy with two fixed key-channel coefficients per head, after the three-coefficient model achieved 99.78%; the next single-channel gauge fix is the narrowest supported reduction.

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
    """Quotient-input QKV map with five reciprocal Q/K scales fixed."""

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
        if head_dim > 2:
            selected_key_channels.add(2)
>>>>>>> REPLACE