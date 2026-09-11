MECHANISM: Normalized-input query-weight gauge fixing

HYPOTHESIS: Anchoring the final input weight of one query projection row will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because LayerNorm inputs lie on an affine hyperplane and the learned query bias can absorb the eliminated constant term.

INTENDED_EDIT: Store seven weights for the first query row, reconstruct its final weight as zero, and initialize the compact row to preserve the function of a full projection on initially zero-mean LayerNorm outputs.

EVIDENCE: The analogous `NormalizedInputLinear` constraint removed the final input weight from all 12 `fc1` rows while achieving 99.93% accuracy at 1,585 parameters; applying it incrementally to one query row tests an independent redundancy after adjacent embedding and `ln1`-bias reductions failed.

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
    """Combined QKV projection with one normalized-input query gauge."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model

        full_weight = self.weight.detach()
        first_query_weight = (
            full_weight[0, :-1] - full_weight[0, -1:]
        )
        self.weight = nn.Parameter(
            torch.cat((first_query_weight, full_weight[1:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-16].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_query_weight = F.pad(
            self.weight[: self.in_features - 1], (0, 1)
        ).unsqueeze(0)
        remaining_weight = self.weight[self.in_features - 1 :].view(
            self.out_features - 1, self.in_features
        )
        weight = torch.cat((first_query_weight, remaining_weight), dim=0)

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
        if isinstance(module, NormalizedInputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(
                    full_weight[:, :-1] - full_weight[:, -1:]
                )
                if module.bias is not None:
                    module.bias.zero_()
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, KeyAnchoredLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            compact_weight = torch.cat(
                (
                    full_weight[0, :-1] - full_weight[0, -1:],
                    full_weight[1:].flatten(),
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.bias.zero_()
        elif isinstance(module, NormalizedInputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(
                    full_weight[:, :-1] - full_weight[:, -1:]
                )
                if module.bias is not None:
                    module.bias.zero_()
>>>>>>> REPLACE