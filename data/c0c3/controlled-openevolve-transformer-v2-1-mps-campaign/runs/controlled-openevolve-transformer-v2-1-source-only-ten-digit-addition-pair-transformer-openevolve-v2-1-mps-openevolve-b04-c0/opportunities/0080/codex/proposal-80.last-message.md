MECHANISM: Shared adaptive attention-input gain

HYPOTHESIS: Sharing `ln1.weight[7]` with learned gain coordinate 1 will reduce the model to 1586 parameters while retaining at least 99% accuracy, because QKV columns can absorb coordinate-specific scaling while the shared value remains trainable.

INTENDED_EDIT: Store only first-LayerNorm gains 1 and 3, and reuse gain 1 for coordinate 7 instead of fixing coordinate 7 at one.

EVIDENCE: Independently fixing coordinate 7 reduced accuracy to 96.28%, while the 1587-parameter design reached 99.63%; sharing preserves adaptive scaling for coordinate 7 while retaining the especially sensitive coordinate 3 independently.

<<<<<<< SEARCH
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gains 0, 2, 4, 5, and 6 and shift 3 fixed."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 5))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))
=======
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gain 7 shared with gain 1 and five gains fixed."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 6))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
                one,
                one,
                one,
                self.weight_rest[2:],
=======
                one,
                one,
                one,
                self.weight_rest[:1],
>>>>>>> REPLACE