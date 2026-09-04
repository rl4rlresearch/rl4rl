MECHANISM: Single-coordinate query-bias gauge fixing

HYPOTHESIS: The qualified four-key-fix model will retain at least 99% accuracy with 1,033 parameters when one query-bias coordinate is fixed at zero, because a per-head query/key change of basis can orient the learned query bias away from that coordinate while preserving attention scores.

INTENDED_EDIT: Apply the qualified symmetric fourth key-coefficient fix and replace the eight-parameter query bias with seven learned coordinates plus one fixed zero coordinate.

EVIDENCE: The four-key-fix design achieved 99.65% accuracy with 1,034 parameters, while a fifth key-coefficient fix collapsed to 0.01%; this motivates testing an orthogonal one-parameter gauge reduction whose fixed query-bias coordinate exactly matches the qualified model’s zero initialization.

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
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
=======
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = QuotientOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + F.pad(self.q_bias, (0, 1))
>>>>>>> REPLACE