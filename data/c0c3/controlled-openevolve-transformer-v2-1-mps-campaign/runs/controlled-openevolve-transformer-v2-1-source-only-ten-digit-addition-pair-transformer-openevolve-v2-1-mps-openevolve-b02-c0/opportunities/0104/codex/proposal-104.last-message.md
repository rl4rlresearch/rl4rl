MECHANISM: Single-key-row normalized-input gauge fixing

HYPOTHESIS: Anchoring one input weight in the first key-projection row will reduce the model to 1,577 parameters while retaining at least 99% accuracy, because `ln1` outputs are zero-mean at initialization and key vectors are invariant to sequence-independent shifts.

INTENDED_EDIT: Store seven coordinates for the first key-projection row, reconstruct its final weight as zero, and initialize from the same full 192-scalar draw while projecting out that row’s common input component.

EVIDENCE: The analogous normalized-input gauge removed one `fc1` weight per row and retained 99.93% accuracy; applying it to only one key row is a more conservative independent reduction than the repeatedly unsuccessful fourth `ln1` scale tie.

<<<<<<< SEARCH
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with tied key and value bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-16].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = self.bias.new_zeros(self.d_model)
        bias = torch.cat(
            (
                self.bias[: self.d_model],
                self.bias.new_zeros(3),
                shared_key_bias,
                self.bias.new_zeros(3),
                shared_key_bias,
                value_bias,
            )
        )
        return F.linear(x, self.weight, bias)
=======
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with anchored biases and one key-row input gauge."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model

        full_weight = self.weight.detach()
        anchored_key_row = (
            full_weight[d_model, :-1] - full_weight[d_model, -1:]
        )
        self.weight = nn.Parameter(
            torch.cat(
                (
                    full_weight[:d_model].flatten(),
                    anchored_key_row,
                    full_weight[d_model + 1 :].flatten(),
                )
            )
        )
        self.bias = nn.Parameter(self.bias.detach()[:-16].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key_row_start = self.d_model * self.d_model
        anchored_key_row = F.pad(
            self.weight[key_row_start : key_row_start + self.d_model - 1],
            (0, 1),
        )
        weight = torch.cat(
            (
                self.weight[:key_row_start],
                anchored_key_row,
                self.weight[key_row_start + self.d_model - 1 :],
            )
        ).view(3 * self.d_model, self.d_model)

        shared_key_bias = self.bias[:1]
        value_bias = self.bias.new_zeros(self.d_model)
        bias = torch.cat(
            (
                self.bias[: self.d_model],
                self.bias.new_zeros(3),
                shared_key_bias,
                self.bias.new_zeros(3),
                shared_key_bias,
                value_bias,
            )
        )
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, OutputAnchoredLinear):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, KeyAnchoredLinear):
            full_weight = module.weight.new_empty(
                3 * module.d_model, module.d_model
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            anchored_key_row = (
                full_weight[module.d_model, :-1]
                - full_weight[module.d_model, -1:]
            )
            compact_weight = torch.cat(
                (
                    full_weight[: module.d_model].flatten(),
                    anchored_key_row,
                    full_weight[module.d_model + 1 :].flatten(),
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.bias.zero_()
        elif isinstance(module, OutputAnchoredLinear):
>>>>>>> REPLACE