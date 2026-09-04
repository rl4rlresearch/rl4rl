MECHANISM: Centered attention-input LayerNorm bias gauge

HYPOTHESIS: Constraining `ln1` bias to seven learned mean-zero contrasts will reduce the model to 1633 parameters while retaining at least 99% accuracy, because all eight effective coordinates remain trainable and the removed common-offset direction is absorbable by query/value biases while its key-bias effect cancels under softmax.

INTENDED_EDIT: Replace the eight independent `ln1` bias parameters with seven parameters that reconstruct eight centered biases, preserving initialization, gains, and all attention/MLP capacity.

EVIDENCE: The balanced `ln2` bias contrast parameterization achieved 99.96% at 1634 parameters, whereas the balanced gain constraint collapsed to 34.34%; this motivates applying the successful distributed bias constraint to `ln1` without altering normalization gains.

<<<<<<< SEARCH
class ReducedBiasLayerNorm(nn.Module):
=======
class CenteredBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        bias = bias - bias.mean()
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)


class ReducedBiasLayerNorm(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = CenteredBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE