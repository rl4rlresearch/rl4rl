MECHANISM: LayerNorm-constrained key-row weight quotient

HYPOTHESIS: Removing one input coefficient from the first key-projection row will reduce the model from 1,523 to 1,522 parameters while retaining at least 99% accuracy, because LayerNorm outputs have one affine constraint and the resulting position-common key offset is softmax-null.

INTENDED_EDIT: Represent the first key row with seven coefficients, reconstruct its final coefficient as zero, and canonicalize initialization by subtracting the removed coefficient from the retained coefficients.

EVIDENCE: The verified model already removes one input coefficient from each of twelve `fc1` rows using the LayerNorm constraint, while its complete removal of key bias confirms that position-common key offsets do not affect attention.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
=======
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


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.qkv = SinglePrunedKeyRowQKVLinear(d_model, 3 * d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight) + full_bias
=======
        qkv = F.linear(x, self.qkv.full_weight()) + full_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, DistributedTwelvePrunedInputWeightLinear):
=======
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
        elif isinstance(module, DistributedTwelvePrunedInputWeightLinear):
>>>>>>> REPLACE