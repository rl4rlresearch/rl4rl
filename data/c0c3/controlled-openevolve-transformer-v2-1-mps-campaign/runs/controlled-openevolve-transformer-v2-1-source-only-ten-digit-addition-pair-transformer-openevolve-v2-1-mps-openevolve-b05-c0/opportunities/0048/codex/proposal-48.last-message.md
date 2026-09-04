MECHANISM: Coordinate-4 LayerNorm-scale absorption into the downstream MLP affine map

HYPOTHESIS: Fixing second-LayerNorm scale coordinate 4 at one will reduce the model from 1621 to 1620 parameters while retaining at least 99% accuracy, because the following full `fc1` weight and bias can exactly absorb that scale degree of freedom.

INTENDED_EDIT: Replace the second LayerNorm with an initialization-preserving variant that learns seven scale coordinates, reconstructs coordinate 4 as one, and retains the full bias.

EVIDENCE: The current coordinate-4 MLP quotient reached 99.93%, while earlier results showed strong coordinate sensitivity. This tests coordinate 4 in a distinct exact downstream-affine quotient after the previous LayerNorm-bias coordinate removal failed.

<<<<<<< SEARCH
        y = self.resid_drop(y)
        return y


class GaugeFixedMLPOutput(nn.Module):
=======
        y = self.resid_drop(y)
        return y


class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale coordinate 4 absorbed by the following affine map."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_index = 4
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight[: self.fixed_index],
                self.weight.new_ones(1),
                self.weight[self.fixed_index :],
            )
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            full_weight,
            self.bias,
            self.eps,
        )


class GaugeFixedMLPOutput(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln2 = GaugeFixedLayerNormScale(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE