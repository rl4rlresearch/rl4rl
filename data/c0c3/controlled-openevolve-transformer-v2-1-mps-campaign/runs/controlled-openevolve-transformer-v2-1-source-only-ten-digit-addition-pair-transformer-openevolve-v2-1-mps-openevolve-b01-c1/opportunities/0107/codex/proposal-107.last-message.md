MECHANISM: Single key-row normalized-input gauge

HYPOTHESIS: Pruning the second input coordinate of key row five will reduce the model from 1,534 to 1,533 parameters while retaining at least 99% accuracy, because LayerNorm’s zero-mean input makes the removed weight an exact row-wise gauge and any bias-induced key offset is softmax-null.

INTENDED_EDIT: Replace QKV with a one-parameter-smaller projection, reconstruct the selected key weight as zero, and canonicalize fresh initialization to preserve its initial attention scores.

EVIDENCE: The current 99.91%-accurate design successfully applies this second-coordinate LayerNorm-input quotient to `fc1` rows five and six. This tests the same optimization-friendly parameterization on a key row, avoiding the distinct orthogonal query/key rotation that reached only 73.92%.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
class SinglePrunedKeyInputLinear(nn.Linear):
    """QKV projection with one normalized-input gauge removed from a key row."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        if out_features != 3 * in_features or in_features <= 5:
            raise ValueError("expected a QKV projection with at least six features")

        self.key_row_index = in_features + 5
        self.input_coordinate = 1
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            torch.cat(
                (
                    full_weight[: self.key_row_index],
                    full_weight[self.key_row_index + 1 :],
                ),
                dim=0,
            ).clone()
        )
        key_row = full_weight[self.key_row_index]
        self.pruned_key_row = nn.Parameter(
            torch.cat(
                (
                    key_row[: self.input_coordinate],
                    key_row[self.input_coordinate + 1 :],
                )
            ).clone()
        )

    def full_weight(self) -> torch.Tensor:
        key_row = torch.cat(
            (
                self.pruned_key_row[: self.input_coordinate],
                self.pruned_key_row.new_zeros(1),
                self.pruned_key_row[self.input_coordinate :],
            )
        ).unsqueeze(0)
        return torch.cat(
            (
                self.weight[: self.key_row_index],
                key_row,
                self.weight[self.key_row_index :],
            ),
            dim=0,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = SinglePrunedKeyInputLinear(d_model, 3 * d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight) + full_bias
=======
        qkv = F.linear(x, self.qkv.full_weight()) + full_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, DistributedTwelvePrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
=======
        elif isinstance(module, SinglePrunedKeyInputLinear):
            # Draw the original full QKV matrix, then remove the selected key
            # row's second coordinate by its zero-mean LayerNorm input gauge.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                row_index = module.key_row_index
                coordinate = module.input_coordinate
                module.weight.copy_(
                    torch.cat(
                        (
                            full[:row_index],
                            full[row_index + 1 :],
                        ),
                        dim=0,
                    )
                )
                canonical_row = full[row_index] - full[row_index, coordinate]
                module.pruned_key_row.copy_(
                    torch.cat(
                        (
                            canonical_row[:coordinate],
                            canonical_row[coordinate + 1 :],
                        )
                    )
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, DistributedTwelvePrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
>>>>>>> REPLACE