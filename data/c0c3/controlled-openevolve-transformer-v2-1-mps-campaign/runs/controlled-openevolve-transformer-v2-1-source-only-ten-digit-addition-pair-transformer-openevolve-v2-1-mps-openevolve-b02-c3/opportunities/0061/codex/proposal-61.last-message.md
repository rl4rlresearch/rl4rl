MECHANISM: Fourth terminal-scale fixing

HYPOTHESIS: The verified 829-parameter transformer will retain at least 99% accuracy with 828 parameters because two learned terminal scales plus the retained three-coordinate final bias should preserve adequate output calibration.

INTENDED_EDIT: Fix one additional classifier-visible terminal LayerNorm scale at one, reducing its learned scales from three to two.

EVIDENCE: The current six-dimensional design achieved 99.88% accuracy, and three consecutive terminal-scale reductions in the eight-dimensional architecture remained above 99%; this adjacent one-parameter reduction extends the most consistently successful compression mechanism without altering the load-bearing five-dimensional lexical representation, seven-unit MLP, or attention routing.

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