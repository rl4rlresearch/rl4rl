MECHANISM: Third first-head value-space shear gauge fixing

HYPOTHESIS: A 823-parameter transformer will retain at least 99% accuracy because fixing a third off-axis coefficient in the tolerant first-head value matrix removes another V/output-projection shear degree without altering attention routing or residual width.

INTENDED_EDIT: Reconstruct the qualified 824-parameter QKV constraints, then additionally fix input coefficient three of the first value channel at zero while leaving the sensitive second head unconstrained.

EVIDENCE: Successive first-head constraints achieved 99.99% at 825 parameters and 99.54% at 824, whereas constraining the second head collapsed; the narrowest informative next reduction is therefore another adjacent shear constraint within the demonstrated tolerant first head.

<<<<<<< SEARCH
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
            for offset in (1, 2, 3)
        }
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