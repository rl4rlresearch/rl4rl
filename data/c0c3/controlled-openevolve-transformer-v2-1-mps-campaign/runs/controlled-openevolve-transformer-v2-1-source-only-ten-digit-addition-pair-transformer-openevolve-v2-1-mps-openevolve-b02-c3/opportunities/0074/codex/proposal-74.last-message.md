MECHANISM: First-head value-basis shear completion

HYPOTHESIS: An 822-parameter transformer will retain at least 99% accuracy because zeroing the remaining off-diagonal coefficient in the first head’s 3×3 value pivot removes another V/output-projection shear degree while keeping both remaining diagonal pivots learnable.

INTENDED_EDIT: Reconstruct the qualified 823-parameter value constraints and additionally fix the third value row’s second input coefficient at zero.

EVIDENCE: The 823-parameter lower-triangular value-basis design achieved 99.83%, while fixing a diagonal coefficient failed; this motivates eliminating the remaining off-diagonal pivot entry without constraining another diagonal or the sensitive second head.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K and all first-head V scales fixed."""

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
            2 * d_model + offset for offset in range(min(3, head_dim))
        )

        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(
            row * in_features for row in sorted(selected_rows)
        )
        self.coeff = nn.Parameter(
            torch.empty(out_features * in_features - len(self.fixed_indices))
        )
        nn.init.normal_(self.coeff, mean=0.0, std=0.02)
        self.register_buffer(
            "fixed_coeff",
            torch.full((len(self.fixed_indices),), 0.02),
            persistent=False,
        )
=======
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K, V scales, and four V shears fixed."""

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
        shear_indices.update(
            {
                (2 * d_model + 1) * in_features + 2,
                (2 * d_model + 2) * in_features + 1,
            }
        )
        selected_indices.update(shear_indices)

        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(sorted(selected_indices))
        self.coeff = nn.Parameter(
            torch.empty(out_features * in_features - len(self.fixed_indices))
        )
        nn.init.normal_(self.coeff, mean=0.0, std=0.02)
        self.register_buffer(
            "fixed_coeff",
            torch.tensor(
                [
                    0.0 if index in shear_indices else 0.02
                    for index in self.fixed_indices
                ]
            ),
            persistent=False,
        )
>>>>>>> REPLACE