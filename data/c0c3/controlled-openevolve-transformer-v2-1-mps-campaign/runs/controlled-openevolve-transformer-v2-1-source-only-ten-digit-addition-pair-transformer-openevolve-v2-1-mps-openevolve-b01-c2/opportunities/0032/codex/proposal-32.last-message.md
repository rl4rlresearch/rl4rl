MECHANISM: Residual-stream common-bias gauge fixing

HYPOTHESIS: Combining the verified affine-free `ln2` and all-row `fc1` gauge with removal of `fc2`’s exact common-mode bias direction will produce a 1,590-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified 1,591-parameter design, then replace `fc2` with an equivalent learned linear layer whose final bias coordinate is fixed at zero.

EVIDENCE: The affine-free `ln2` plus all-row `fc1` design achieved 99.96% at 1,591 parameters, while the broad 1,560-parameter `ln1`/QKV reduction failed; this tests one orthogonal null direction whose zero-bias initialization leaves the initial function unchanged.

<<<<<<< SEARCH
        self.gauged_rows = (0, 1, 2, 4, 5, 6)
=======
        self.gauged_rows = tuple(range(out_features))
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight = torch.stack(rows)
        return F.linear(x, weight, self.bias)


class MLP(nn.Module):
=======
        weight = torch.stack(rows)
        return F.linear(x, weight, self.bias)


class ResidualBiasGaugedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        # Preserve the replaced nn.Linear constructor RNG stream.
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.new_empty(out_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A common shift of the residual stream is removed by subsequent
        # LayerNorms, so every bias equivalence class has a last-zero member.
        bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = ResidualBiasGaugedLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = AnchoredScaleLayerNorm(cfg.d_model)
=======
        # fc1 absorbs feature scales; affine-free normalization also makes all
        # fc1 common-row coefficients exact null directions during training.
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE