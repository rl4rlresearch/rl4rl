MECHANISM: Per-head Q/K shear-gauge fixing

HYPOTHESIS: An 827-parameter transformer will retain at least 99% accuracy because fixing one off-axis key coefficient per head removes Q/K channel-basis shear degrees while preserving the verified rank-five lexical state, seven-unit MLP, three terminal scales, and unconstrained final key-channel scales.

INTENDED_EDIT: Reconstruct the verified 829-parameter six-dimensional design, then fix one additional key-matrix coefficient at zero in each head, removing two learned parameters.

EVIDENCE: The 829-parameter design achieved 99.88% accuracy; the 827-parameter full key-scale fixing reached only 96.24%, motivating a different gauge reduction that leaves the remaining key-channel scales learnable.

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
    """Quotient-input QKV map with scale and shear gauges fixed per head."""

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
        fixed_coefficients = {
            (d_model + channel) * in_features: 0.02
            for channel in selected_key_channels
        }
        for head in range(n_head):
            fixed_coefficients[
                (d_model + head * head_dim) * in_features + 1
            ] = 0.0

        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(sorted(fixed_coefficients))
        self.coeff = nn.Parameter(
            torch.empty(out_features * in_features - len(self.fixed_indices))
        )
        nn.init.normal_(self.coeff, mean=0.0, std=0.02)
        self.register_buffer(
            "fixed_coeff",
            torch.tensor(
                [fixed_coefficients[index] for index in self.fixed_indices]
            ),
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class ClassifierAwareLayerNorm(nn.Module):
    """LayerNorm with two visible scales and ignored scales fixed."""

    def __init__(self, normalized_shape: int, relevant_features: int):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(relevant_features - 2))
        self.register_buffer(
            "fixed_weight",
            torch.ones(normalized_shape - relevant_features + 2),
            persistent=False,
        )
=======
class ClassifierAwareLayerNorm(nn.Module):
    """LayerNorm with three learned scales and remaining scales fixed."""

    def __init__(self, normalized_shape: int, relevant_features: int):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(relevant_features - 3))
        self.register_buffer(
            "fixed_weight",
            torch.ones(normalized_shape - relevant_features + 3),
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 3
        )
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-model", type=int, default=8)
=======
    p.add_argument("--d-model", type=int, default=6)
>>>>>>> REPLACE