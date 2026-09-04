MECHANISM: Per-head value/output scale gauge fixing

HYPOTHESIS: A 827-parameter transformer will retain at least 99% accuracy because fixing the first value-channel coefficient in the second attention head extends the value/output reciprocal scale gauge that achieved 99.97% at 828 parameters without changing attention routing or representational width.

INTENDED_EDIT: Fix the first coefficient of one value channel in each attention head, removing one additional learned parameter from the verified 828-parameter design.

EVIDENCE: The current 828-parameter model achieved 99.97% after one value-channel coefficient was fixed, while terminal-scale and additional Q/K reductions failed; this makes the corresponding value-scale gauge in the other head the narrowest evidence-backed next reduction.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K and one V/output scale fixed."""

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
        selected_rows.add(2 * d_model)
=======
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K and per-head V scales fixed."""

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
            2 * d_model + head * head_dim for head in range(n_head)
        )
>>>>>>> REPLACE