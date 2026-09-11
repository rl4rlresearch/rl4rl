MECHANISM: Final-normalization scale fixing

HYPOTHESIS: A 671-parameter transformer will retain at least 99% accuracy because it preserves the qualified 672-parameter attention, lexical, and MLP architecture while fixing only one final LayerNorm scale at its unit initialization.

INTENDED_EDIT: Reconstruct the qualified four-coordinate MLP bias fixing, then remove one learned final-normalization scale by extending the unit-valued fixed-scale suffix from three features to four.

EVIDENCE: The 672-parameter design achieved 99.52%, while removing its remaining MLP output bias collapsed to 33.33% and tying a lexical coordinate collapsed to 56.63%; this motivates preserving those components and testing an isolated reduction in the previously untested final normalization.

<<<<<<< SEARCH
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=3
        )
=======
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=4
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class ClassifierAwareLayerNorm(nn.Module):
    """LayerNorm with three visible scales and remaining scales fixed."""

    def __init__(self, normalized_shape: int, relevant_features: int):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(relevant_features - 3))
        self.register_buffer(
            "fixed_weight",
            torch.ones(normalized_shape - relevant_features + 3),
            persistent=False,
        )
=======
class ClassifierAwareLayerNorm(nn.Module):
    """LayerNorm with two visible scales and remaining scales fixed."""

    def __init__(self, normalized_shape: int, relevant_features: int):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(relevant_features - 4))
        self.register_buffer(
            "fixed_weight",
            torch.ones(normalized_shape - relevant_features + 4),
            persistent=False,
        )
>>>>>>> REPLACE