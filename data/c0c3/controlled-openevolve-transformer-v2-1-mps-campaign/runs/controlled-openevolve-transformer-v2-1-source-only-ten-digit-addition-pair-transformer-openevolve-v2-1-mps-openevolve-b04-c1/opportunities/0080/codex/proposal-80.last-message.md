MECHANISM: Tied feature-dictionary feed-forward readout

HYPOTHESIS: Reusing the 12 learned MLP feature directions as output directions, with one learned gain per feature, will reduce the model from 965 to 893 parameters while retaining at least 99% accuracy because attention and the load-bearing four-channel token representation remain unchanged.

INTENDED_EDIT: Replace the independent 84-parameter MLP output matrix with the transpose of `fc1` and 12 learned channel gains, preserve the verified initialization random stream, and remove obsolete MLP gauge-optimizer entries.

EVIDENCE: The four-channel token bottleneck retained 99.84% accuracy while three channels fell to 96.91%, motivating preservation of that interface while challenging the separate assumption that the MLP requires independently learned analysis and synthesis matrices.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = OneColumnGaugeLinear(d_ff, d_model)
        self.output_bias = ShiftGaugeBias(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))) + self.output_bias())
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.channel_gain = nn.Parameter(torch.ones(d_ff))

        # Preserve the constructor-time draws of the replaced gauge-linear
        # decoder so the attention and feature dictionary retain their
        # verified apply-time initialization stream.
        conceptual_fc2_rest = torch.empty(d_model, d_ff - 2)
        nn.init.kaiming_uniform_(conceptual_fc2_rest, a=math.sqrt(5))
        conceptual_fc2_first = torch.empty(2, d_model - 1)
        nn.init.normal_(conceptual_fc2_first, mean=0.0, std=0.02)

        self.legacy_fc2_reset_shape = (d_model, d_ff - 2)
        self.output_bias = ShiftGaugeBias(d_model)
        self.drop = nn.Dropout(dropout)

    def consume_legacy_fc2_reset_draw(self) -> None:
        conceptual_fc2_rest = self.fc1.weight.new_empty(
            self.legacy_fc2_reset_shape
        )
        nn.init.normal_(conceptual_fc2_rest, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = F.gelu(self.fc1(x)) * self.channel_gain
        decoded = F.linear(features, self.fc1.weight.transpose(0, 1))
        return self.drop(decoded + self.output_bias())
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, OneRowInputGaugeLinear):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
        elif isinstance(module, OneRowInputGaugeLinear):
            module.reset_parameters()
        elif isinstance(module, MLP):
            module.consume_legacy_fc2_reset_draw()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_entries = [
        (
            coordinates,
            block.mlp.fc2.reflector,
            block.mlp.fc2.reflector_norm_sq,
        )
        for block in model.blocks
        for coordinates in (
            block.mlp.fc2.third_coordinates,
            block.mlp.fc2.fourth_coordinates,
            block.mlp.fc2.fifth_coordinates,
            block.mlp.fc2.sixth_coordinates,
            block.mlp.fc2.seventh_coordinates,
            block.mlp.fc2.eighth_coordinates,
            block.mlp.fc2.ninth_coordinates,
            block.mlp.fc2.tenth_coordinates,
            block.mlp.fc2.middle_coordinates,
        )
    ] + [
=======
    gauge_entries = [
>>>>>>> REPLACE