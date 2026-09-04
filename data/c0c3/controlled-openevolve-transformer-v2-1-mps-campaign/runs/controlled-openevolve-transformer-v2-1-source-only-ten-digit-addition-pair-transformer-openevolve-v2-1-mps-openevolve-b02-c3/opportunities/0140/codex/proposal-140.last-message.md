MECHANISM: Canonical fixed value frame with learned output decoding

HYPOTHESIS: A 661-parameter transformer will retain at least 99% accuracy because the first head’s V/output GL(3) symmetry allows its remaining three learned value-frame coefficients to be fixed as a full-rank canonical frame while the learned output projection absorbs the basis change.

INTENDED_EDIT: Complete the partially fixed first-head 3×3 value frame by fixing its two remaining diagonal coefficients at 0.02 and its remaining off-diagonal coefficient at zero, removing exactly three learned QKV parameters without compressing routing biases or MLP capacity.

EVIDENCE: The current 664-parameter model achieved 99.57%, and the 663-parameter first-head Q/K gauge fixing achieved 99.88%, showing that attention-coordinate redundancies can be removed safely. Conversely, tying functionally distinct MLP directions collapsed to 22.53%, motivating an exact V/output basis gauge rather than another functional-capacity tie.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K, V scales, and three V shears fixed."""

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
        selected_indices = {
            (d_model + channel) * in_features
            for channel in selected_key_channels
        }
        selected_indices.update(
            (2 * d_model + offset) * in_features
            for offset in range(min(3, head_dim))
        )
        shear_indices = {
            (2 * d_model) * in_features + offset
            for offset in (1, 2)
        }
        shear_indices.add(
            (2 * d_model + 1) * in_features + 2
        )
        selected_indices.update(shear_indices)
=======
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with a canonical first-head value frame."""

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
        selected_indices = {
            (d_model + channel) * in_features
            for channel in selected_key_channels
        }
        selected_indices.update(
            (2 * d_model + offset) * in_features
            for offset in range(min(3, head_dim))
        )
        selected_indices.update(
            (2 * d_model + offset) * in_features + offset
            for offset in range(1, min(3, head_dim))
        )
        shear_indices = {
            (2 * d_model) * in_features + offset
            for offset in (1, 2)
        }
        shear_indices.add(
            (2 * d_model + 1) * in_features + 2
        )
        shear_indices.add(
            (2 * d_model + 2) * in_features + 1
        )
        selected_indices.update(shear_indices)
>>>>>>> REPLACE