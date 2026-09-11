MECHANISM: Single-row LayerNorm nullspace gauge fixing

HYPOTHESIS: Gauge-fixing only one QKV input-weight row will reduce the model from 1,596 to 1,595 parameters while retaining at least 99% accuracy, because non-affine LayerNorm makes one weight direction unobservable and gauge-equivalent initialization limits optimization disruption to a single row.

INTENDED_EDIT: Replace the QKV linear map with an equivalent learned map whose first output row uses seven anchored weights while its other 23 rows remain unchanged.

EVIDENCE: Compressing all 36 nullspace directions reached 97.76%, close to the threshold, while the 1,596-parameter model reached 99.99%; testing one of those exact redundancies is the smallest informative step toward locating the safe compression boundary.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class AnchoredMeanZeroRowLinear(nn.Module):
    """Linear map with one input-weight row gauge-fixed for mean-zero inputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.anchored_weight = nn.Parameter(torch.empty(in_features - 1))
        self.remaining_weight = nn.Parameter(torch.empty(out_features - 1, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        full = self.anchored_weight.new_empty(self.out_features, self.in_features)
        nn.init.normal_(full, mean=0.0, std=0.02)
        with torch.no_grad():
            self.anchored_weight.copy_(full[0, :-1] - full[0, -1])
            self.remaining_weight.copy_(full[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = F.linear(x[..., :-1], self.anchored_weight.unsqueeze(0))
        remaining = F.linear(x, self.remaining_weight)
        return torch.cat((first, remaining), dim=-1)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = AnchoredMeanZeroRowLinear(d_model, 3 * d_model)
>>>>>>> REPLACE