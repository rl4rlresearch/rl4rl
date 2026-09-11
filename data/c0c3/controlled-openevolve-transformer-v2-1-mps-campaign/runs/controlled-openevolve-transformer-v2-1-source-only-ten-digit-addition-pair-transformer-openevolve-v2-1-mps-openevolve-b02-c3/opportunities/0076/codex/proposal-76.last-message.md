MECHANISM: Cross-head sharing of the sparsest relative-distance bias

HYPOTHESIS: An 822-parameter transformer will retain at least 99% accuracy because it preserves the qualified 823-parameter value path while sharing only the learned maximum-distance bias, which affects a single causal query-key pair per head at full sequence length.

INTENDED_EDIT: Reconstruct the qualified 823-parameter QKV constraints, then replace the two head-specific farthest-distance biases with one shared learned scalar, removing one parameter.

EVIDENCE: The 823-parameter lower-triangular value design achieved 99.83%; both failed 822-parameter attempts further constrained the value/output path, motivating an orthogonal reduction at the sparsest relative-attention distance.

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
        # Each head learns routing by causal relative distance. Distance zero
        # is fixed because a per-head common bias cancels inside softmax.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
=======
        # Distance zero is fixed because a per-head common bias cancels inside
        # softmax. The sparsely used maximum distance shares one learned bias.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 2))
        self.farthest_relative_bias = nn.Parameter(torch.zeros(1))
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
                self.farthest_relative_bias.expand(self.n_head, 1),
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