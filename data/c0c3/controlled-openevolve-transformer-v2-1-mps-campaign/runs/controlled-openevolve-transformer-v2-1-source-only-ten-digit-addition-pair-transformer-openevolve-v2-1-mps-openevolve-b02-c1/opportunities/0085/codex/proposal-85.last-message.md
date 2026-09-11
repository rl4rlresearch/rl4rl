MECHANISM: Terminal-paired scalar MLP input

HYPOTHESIS: Sharing the final two effective MLP input coefficients will reduce the model from 999 to 998 parameters while retaining at least 99% accuracy, because removing the scalar MLP offset retained 99.98% and this preserves the independently learned output routing that failed under terminal `fc2` sharing.

INTENDED_EDIT: Replace `fc1` with a six-parameter input projection that reuses its final coefficient for the seventh input channel, while preserving the original constructor and initialization RNG streams.

EVIDENCE: Fixing the width-one MLP bias at zero retained 99.98%, whereas sharing the final two MLP output contrasts collapsed to 25.76%; this motivates compressing the scalar neuron’s input direction while leaving its sensitive output vector unchanged.

<<<<<<< SEARCH
class MLP(nn.Module):
=======
class TerminalPairedMLPInput(nn.Module):
    """Scalar MLP input with its final two coefficients shared."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features - 1)
        )

        # Preserve the RNG stream of the replaced bias-free Linear.
        discarded_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(discarded_weight, a=math.sqrt(5))

    def initialize_from_full_normal(self) -> None:
        full_weight = self.weight.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.weight.copy_(full_weight[:, :-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            [self.weight, self.weight[:, -1:]], dim=-1
        )
        return F.linear(x, weight)


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model - 1, d_ff, bias=False)
=======
        self.fc1 = TerminalPairedMLPInput(d_model - 1, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedQKV):
            module.initialize_from_full_normal()
        elif isinstance(module, GaugeFixedEmbedding):
=======
        if isinstance(module, GaugeFixedQKV):
            module.initialize_from_full_normal()
        elif isinstance(module, TerminalPairedMLPInput):
            module.initialize_from_full_normal()
        elif isinstance(module, GaugeFixedEmbedding):
>>>>>>> REPLACE