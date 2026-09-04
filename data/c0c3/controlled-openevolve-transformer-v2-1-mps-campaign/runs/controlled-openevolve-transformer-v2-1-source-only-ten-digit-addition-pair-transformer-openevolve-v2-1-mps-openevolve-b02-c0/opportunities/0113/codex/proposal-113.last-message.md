MECHANISM: Scale-aware key-row gauge fixing

HYPOTHESIS: Removing one key-projection parameter with a gauge that tracks the learned `ln1` scales will produce a 1,575-parameter model with at least 99% accuracy, because it preserves the key-shift invariance throughout training rather than only at initialization.

INTENDED_EDIT: Compact the first key-projection row to seven coordinates, dynamically reconstruct its eighth coordinate orthogonal to the inverse LayerNorm scale, and initialize it from the same full 192-weight draw projected onto that gauge.

EVIDENCE: The previous fixed-coordinate key-row reduction reached only 53.22% and relied on `ln1` outputs being zero-mean at initialization; learned unequal scales invalidate that premise. The current 1,576-parameter model reaches 99.92%, so correcting that specific gauge mismatch is an informative one-parameter reduction.

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
    """Combined QKV projection with a scale-aware key-row gauge."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model

        full_weight = self.weight.detach().flatten()
        self.key_gauge_flat_index = d_model * d_model + d_model - 1
        self.weight = nn.Parameter(
            torch.cat(
                (
                    full_weight[: self.key_gauge_flat_index],
                    full_weight[self.key_gauge_flat_index + 1 :],
                )
            )
        )
        self.bias = nn.Parameter(self.bias.detach()[:-16].clone())

    def forward(
        self, x: torch.Tensor, input_scale: torch.Tensor
    ) -> torch.Tensor:
        key_row_start = self.key_gauge_flat_index - (self.d_model - 1)
        key_row_prefix = self.weight[
            key_row_start : self.key_gauge_flat_index
        ]
        key_row_final = -input_scale[-1] * torch.sum(
            key_row_prefix / input_scale[:-1]
        )
        full_weight = torch.cat(
            (
                self.weight[: self.key_gauge_flat_index],
                key_row_final.reshape(1),
                self.weight[self.key_gauge_flat_index :],
            )
        ).view(self.out_features, self.in_features)

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
        return F.linear(x, full_weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
=======
    def forward(
        self, x: torch.Tensor, input_scale: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x, input_scale)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight,
                self.weight[1:2],
                self.weight[2:3],
                self.weight[:2],
            )
        )
        bias = F.pad(self.bias, (0, 3))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
=======
    def full_weight(self) -> torch.Tensor:
        return torch.cat(
            (
                self.weight,
                self.weight[1:2],
                self.weight[2:3],
                self.weight[:2],
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 3))
        return F.layer_norm(
            x, (x.size(-1),), self.full_weight(), bias, 1e-5
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.ln1(x)
        x = x + self.attn(normalized, self.ln1.full_weight())
        x = x + self.mlp(self.ln2(x))
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, OutputAnchoredLinear):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, KeyAnchoredLinear):
            baseline_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(baseline_weight, mean=0.0, std=0.02)
            baseline_weight[module.d_model].sub_(
                baseline_weight[module.d_model].mean()
            )
            flat_weight = baseline_weight.flatten()
            compact_weight = torch.cat(
                (
                    flat_weight[: module.key_gauge_flat_index],
                    flat_weight[module.key_gauge_flat_index + 1 :],
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.bias.zero_()
        elif isinstance(module, OutputAnchoredLinear):
>>>>>>> REPLACE