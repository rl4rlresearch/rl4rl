MECHANISM: Redundant value/output-bias coordinate tying

HYPOTHESIS: Tying one attention output-bias coordinate to a learned value-bias coordinate will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because value bias and projection bias contribute only through their combined position-independent projected offset.

INTENDED_EDIT: Keep all eight learned value-bias coordinates, but reuse the final value-bias scalar as the final mean-free attention projection-bias coordinate instead of learning that coordinate independently.

EVIDENCE: Removing even one value-bias coordinate reduced accuracy to 97.31%, while the full-value-bias design reached 99.62% at 1,584 parameters; tying a redundant projection-bias coordinate preserves every value channel and its direct learned gradient path.

<<<<<<< SEARCH
        self.proj = MeanFreeResidualLinear(d_model, d_model)
=======
        self.proj = SharedBiasMeanFreeResidualLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
=======
        y = self.proj(y, v_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        return F.linear(x, full_weight, full_bias)


class MLP(nn.Module):
=======
        return F.linear(x, full_weight, full_bias)


class SharedBiasMeanFreeResidualLinear(MeanFreeResidualLinear):
    """Mean-free projection sharing one redundant bias with its value input."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-1].clone())

    def forward(
        self, x: torch.Tensor, shared_bias: torch.Tensor
    ) -> torch.Tensor:
        full_weight = self.bias_basis @ self.weight
        bias_coordinates = torch.cat((self.bias, shared_bias.reshape(1)))
        full_bias = bias_coordinates @ self.bias_basis.transpose(0, 1)
        return F.linear(x, full_weight, full_bias)


class MLP(nn.Module):
>>>>>>> REPLACE