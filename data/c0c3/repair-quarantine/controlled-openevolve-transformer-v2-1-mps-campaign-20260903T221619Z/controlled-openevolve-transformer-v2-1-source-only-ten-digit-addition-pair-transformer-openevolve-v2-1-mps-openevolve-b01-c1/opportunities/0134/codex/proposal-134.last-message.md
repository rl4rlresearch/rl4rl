MECHANISM: LayerNorm-constrained second key-row weight quotient

HYPOTHESIS: Removing one input coefficient from the second key-projection row will reduce the verified model from 1,522 to 1,521 parameters while retaining at least 99% accuracy, because the same LayerNorm constraint and softmax-null key offset that supported pruning the first key row apply independently to every key row.

INTENDED_EDIT: Represent the first two key-projection rows with seven coefficients each, reconstruct their final coefficients as zero, and canonicalize both rows during initialization.

EVIDENCE: Pruning one coefficient from the first key-projection row achieved 99.95% accuracy at 1,522 parameters; extending that identical successful quotient to the adjacent independent key row is the smallest informative reduction.

<<<<<<< SEARCH
class SinglePrunedKeyRowQKVLinear(nn.Linear):
    """QKV map with one LayerNorm-redundant key-row coefficient removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            torch.cat(
                (
                    full_weight[:in_features],
                    full_weight[in_features + 1 :],
                ),
                dim=0,
            ).clone()
        )
        self.first_key_row = nn.Parameter(
            full_weight[in_features, :-1].clone()
        )

    def full_weight(self) -> torch.Tensor:
        first_key_row = F.pad(self.first_key_row, (0, 1)).unsqueeze(0)
        return torch.cat(
            (
                self.weight[: self.in_features],
                first_key_row,
                self.weight[self.in_features :],
            ),
            dim=0,
        )
=======
class TwoPrunedKeyRowsQKVLinear(nn.Linear):
    """QKV map with one LayerNorm-redundant coefficient per first two key rows."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            torch.cat(
                (
                    full_weight[:in_features],
                    full_weight[in_features + 2 :],
                ),
                dim=0,
            ).clone()
        )
        self.first_two_key_rows = nn.Parameter(
            full_weight[in_features : in_features + 2, :-1].clone()
        )

    def full_weight(self) -> torch.Tensor:
        first_two_key_rows = F.pad(self.first_two_key_rows, (0, 1))
        return torch.cat(
            (
                self.weight[: self.in_features],
                first_two_key_rows,
                self.weight[self.in_features :],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = SinglePrunedKeyRowQKVLinear(d_model, 3 * d_model)
=======
        self.qkv = TwoPrunedKeyRowsQKVLinear(d_model, 3 * d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, SinglePrunedKeyRowQKVLinear):
            # Preserve the original dense QKV draw, then quotient the first key
            # row by the zero-mean constraint of the initial LayerNorm output.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(
                    torch.cat(
                        (
                            full[: module.in_features],
                            full[module.in_features + 1 :],
                        ),
                        dim=0,
                    )
                )
                module.first_key_row.copy_(
                    full[module.in_features, :-1]
                    - full[module.in_features, -1]
                )
                nn.init.zeros_(module.bias)
=======
        elif isinstance(module, TwoPrunedKeyRowsQKVLinear):
            # Preserve the original dense QKV draw, then quotient the first two
            # key rows by the LayerNorm constraint.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(
                    torch.cat(
                        (
                            full[: module.in_features],
                            full[module.in_features + 2 :],
                        ),
                        dim=0,
                    )
                )
                module.first_two_key_rows.copy_(
                    full[module.in_features : module.in_features + 2, :-1]
                    - full[
                        module.in_features : module.in_features + 2, -1:
                    ]
                )
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE