MECHANISM: Single-channel value/output projection scale gauge

HYPOTHESIS: An 828-parameter transformer will retain at least 99% accuracy because fixing one value-channel coefficient removes a reciprocal V/output-projection scale degree without changing attention routing or reducing the qualified lexical, MLP, and terminal calibration capacity.

INTENDED_EDIT: Reconstruct the verified 829-parameter six-dimensional design, then fix the first coefficient of one value channel at its initialization scale, removing one learned parameter whose magnitude can be absorbed by the corresponding output-projection column.

EVIDENCE: The six-dimensional 829-parameter design achieved 99.88%; terminal-scale removal collapsed to 8.21% and fixing both remaining Q/K scales reached 96.24%, motivating a one-parameter gauge reduction in the value path that preserves both terminal calibration and attention scores.

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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
class ClassifierAwareLayerNorm(nn.Module):
    """LayerNorm with one global scale gauge and ignored scales fixed."""

    def __init__(self, normalized_shape: int, relevant_features: int):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(relevant_features - 1))
        self.register_buffer(
            "fixed_weight",
            torch.ones(normalized_shape - relevant_features + 1),
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