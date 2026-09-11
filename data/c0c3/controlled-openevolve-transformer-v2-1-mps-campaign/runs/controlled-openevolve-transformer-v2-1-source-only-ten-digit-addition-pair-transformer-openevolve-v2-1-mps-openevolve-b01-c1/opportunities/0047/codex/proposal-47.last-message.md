MECHANISM: Incremental LayerNorm-beta/downstream-bias gauge reorientation

HYPOTHESIS: Removing a second `fc1` bias scalar will reduce the model from 1,578 to 1,577 parameters while retaining at least 99% accuracy, because the preceding `ln2` retains six learned bias coordinates that can supply both omitted neuron offsets through `fc1`’s learned weight rows.

INTENDED_EDIT: Replace the one-pruned MLP input linear layer with a two-pruned version that fixes its final two output-bias coordinates at zero while preserving its full weight matrix and all other model and training settings.

EVIDENCE: Removing the first `fc1` bias scalar achieved 99.92% accuracy at 1,578 parameters, whereas pruning an additional `ln2` bias coordinate collapsed accuracy; this directly supports extending the successful downstream-bias removal while preserving the LayerNorm optimization pathway.

<<<<<<< SEARCH
class OnePrunedOutputBiasLinear(nn.Linear):
    """Linear layer with one bias scalar absorbed by the preceding LayerNorm bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 1))
        return F.linear(x, self.weight, bias)
=======
class TwoPrunedOutputBiasLinear(nn.Linear):
    """Linear layer with two bias scalars absorbed by the preceding LayerNorm bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-2].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 2))
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = OnePrunedOutputBiasLinear(d_model, d_ff)
=======
        self.fc1 = TwoPrunedOutputBiasLinear(d_model, d_ff)
>>>>>>> REPLACE