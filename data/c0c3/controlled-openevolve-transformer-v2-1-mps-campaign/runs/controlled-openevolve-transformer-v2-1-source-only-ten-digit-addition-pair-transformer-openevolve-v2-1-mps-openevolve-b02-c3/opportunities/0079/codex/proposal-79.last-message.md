MECHANISM: Fixed maximum-distance attention bias

HYPOTHESIS: An 821-parameter transformer will retain at least 99% accuracy because fixing the already-shared maximum-distance bias at zero changes only the single sparsest causal query-key pair per head while preserving independent learned biases at every nearer distance.

INTENDED_EDIT: Reconstruct the qualified 822-parameter value-path constraints and replace its shared learned farthest-distance bias with a fixed zero.

EVIDENCE: The 822-parameter design reached 99.94% when only the farthest-distance bias was shared, whereas sharing the adjacent distance collapsed to 5.46%; this isolates the next reduction to the successful maximum-distance mechanism without constraining the important neighboring bias.

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

<<<<<<< SEARCH
        self.proj = QuotientOutputLinear(d_model, d_model)
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = QuotientOutputLinear(d_model, d_model)

        # Distance zero cancels inside softmax, and the uniquely sparse
        # maximum distance is fixed at zero for both heads.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 2))

        self.attn_drop = nn.Dropout(dropout)
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
        learned_relative_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 1),
            ),
            dim=1,
        )
        relative_bias = torch.cat(
            (
                learned_relative_bias.new_zeros(self.n_head, 1),
                learned_relative_bias[:, : seqlen - 1],
            ),
            dim=1,
        )
>>>>>>> REPLACE