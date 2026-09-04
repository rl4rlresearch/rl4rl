MECHANISM: Single fixed final-normalization scale

HYPOTHESIS: A 800-parameter transformer will retain at least 99% accuracy because fixing the third classifier-relevant LayerNorm scale at one preserves two learned scales, the full-rank tied lexical embedding, attention routing, and all four gated nonlinear features.

INTENDED_EDIT: Learn only the first two final LayerNorm scales and fix the remaining four scales at one, removing one parameter.

EVIDENCE: The 801-parameter design achieved 99.51% accuracy after eliminating the classifier offset; its remaining margin motivates the narrowest one-parameter reduction while avoiding the lexical-rank reduction that fell to 41.97%.

<<<<<<< SEARCH
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
=======
class ClassifierAwareLayerNorm(nn.Module):
    """LayerNorm with two learned scales and remaining scales fixed."""

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