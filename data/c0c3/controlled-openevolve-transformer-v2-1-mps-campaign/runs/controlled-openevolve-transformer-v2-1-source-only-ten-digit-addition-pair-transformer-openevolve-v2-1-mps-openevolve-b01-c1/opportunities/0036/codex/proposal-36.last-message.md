MECHANISM: Incremental final-output bias absorption

HYPOTHESIS: Removing one additional final-LayerNorm bias direction will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because token-row means provide a learned output-bias channel while remaining invisible to the input-side LayerNorm.

INTENDED_EDIT: Reduce the learned final-LayerNorm bias basis from six to five coordinates, preserving all attention, MLP, value-bias, and training settings.

EVIDENCE: The existing two-direction final-LayerNorm quotient achieved 99.62% at 1,584 parameters. Unlike the failed third `ln2` bias removal, this extends the demonstrated output-side absorption mechanism where tied-token row means directly replace omitted logit offsets.

<<<<<<< SEARCH
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with two bias directions absorbed by token-row means."""
=======
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with three bias directions absorbed by token-row means."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        basis = torch.zeros(normalized_shape, normalized_shape - 2)
        for j in range(normalized_shape - 2):
=======
        basis = torch.zeros(normalized_shape, normalized_shape - 3)
        for j in range(normalized_shape - 3):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 2))
=======
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 3))
>>>>>>> REPLACE