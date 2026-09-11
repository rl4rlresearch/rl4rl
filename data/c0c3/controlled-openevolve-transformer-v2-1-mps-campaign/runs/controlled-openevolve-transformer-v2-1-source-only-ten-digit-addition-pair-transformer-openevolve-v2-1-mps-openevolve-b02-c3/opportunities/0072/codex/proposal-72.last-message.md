MECHANISM: Structured first-head value-basis pivot fixing

HYPOTHESIS: A 823-parameter transformer will retain at least 99% accuracy because fixing the second coefficient of the second first-head value channel completes another entry of an invertible 3×3 value-basis pivot, using a V/output-projection gauge degree without constraining non-pivot value features.

INTENDED_EDIT: Reconstruct the qualified 824-parameter first-head value constraints and additionally fix the second value row’s second input coefficient at 0.02.

EVIDENCE: The 824-parameter design achieved 99.54%, while the failed 823-parameter design fixed a fourth-column coefficient outside the three-column value-basis pivot; this tests a structurally distinct, gauge-aligned reduction within the tolerant first head.

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
    """QKV map with a structured first-head value-basis pivot fixed."""

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
        zero_indices = {
            (2 * d_model) * in_features + offset
            for offset in (1, 2)
        }
        selected_indices.update(zero_indices)
        selected_indices.add((2 * d_model + 1) * in_features + 1)

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
                    0.0 if index in zero_indices else 0.02
                    for index in self.fixed_indices
                ]
            ),
            persistent=False,
        )
>>>>>>> REPLACE