MECHANISM: Argmax-invariant terminal scale gauge

HYPOTHESIS: Fixing one classifier-visible terminal LayerNorm scale at one will reduce the model to 969 parameters while retaining at least 99% accuracy, because the remaining scales and final latent bias can represent the original terminal vector up to a positive global logit scale, which does not change greedy decoding.

INTENDED_EDIT: Learn five terminal LayerNorm scales and fix the sixth classifier-visible scale alongside the two classifier-ignored scales.

EVIDENCE: The 970-parameter design achieved 99.61% accuracy, while removing a third final-bias coordinate collapsed to 45.45%; this preserves all successful output-bias capacity and instead removes a terminal scale degree associated with global logit magnitude.

<<<<<<< SEARCH
class ClassifierAwareLayerNorm(nn.Module):
    """LayerNorm with fixed scales on coordinates ignored by the classifier."""

    def __init__(self, normalized_shape: int, relevant_features: int):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(relevant_features))
        self.register_buffer(
            "fixed_weight",
            torch.ones(normalized_shape - relevant_features),
            persistent=False,
        )
=======
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
>>>>>>> REPLACE