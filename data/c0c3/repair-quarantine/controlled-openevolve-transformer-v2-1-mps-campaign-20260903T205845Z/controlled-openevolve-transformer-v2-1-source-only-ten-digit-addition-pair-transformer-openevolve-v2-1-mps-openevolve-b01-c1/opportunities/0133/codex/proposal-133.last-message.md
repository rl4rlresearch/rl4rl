MECHANISM: Single-key-row LayerNorm-tangent quotient

HYPOTHESIS: Removing one centered-input coefficient from the first key-projection row will reduce the model from 1,525 to 1,524 parameters while retaining at least 99% accuracy, because the row’s omitted coefficient is representationally redundant on centered LayerNorm states and its constant key contribution is softmax-null.

INTENDED_EDIT: Store the QKV matrix with the final input coefficient of the first key row omitted, reconstruct it as zero during inference, and canonicalize that row at initialization while preserving the original RNG sequence.

EVIDENCE: Simultaneously pruning one coefficient from all 24 QKV rows collapsed accuracy to 40.14%, while the current unpruned-QKV design reaches 99.88%; isolating the quotient to one key row tests whether the failure came from applying 24 optimization-changing reparameterizations at once.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class SinglePrunedKeyInputLinear(nn.Linear):
    """QKV map with one centered-input coefficient removed from one key row."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.pruned_index = in_features * in_features + (in_features - 1)
        flat_weight = self.weight.detach().reshape(-1)
        self.weight = nn.Parameter(
            torch.cat(
                (
                    flat_weight[: self.pruned_index],
                    flat_weight[self.pruned_index + 1 :],
                )
            ).clone()
        )

    def full_weight(self) -> torch.Tensor:
        flat_weight = torch.cat(
            (
                self.weight[: self.pruned_index],
                self.weight.new_zeros(1),
                self.weight[self.pruned_index :],
            )
        )
        return flat_weight.view(self.out_features, self.in_features)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.qkv = SinglePrunedKeyInputLinear(d_model, 3 * d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight) + full_bias
=======
        qkv = F.linear(x, self.qkv.full_weight()) + full_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanFreeResidualLinear):
=======
        elif isinstance(module, SinglePrunedKeyInputLinear):
            # Draw the original full QKV matrix to preserve the RNG sequence.
            # On initially centered LayerNorm states, subtracting the omitted
            # coefficient from the rest of its key row preserves that row's
            # input-dependent output; any constant key shift is softmax-null.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                key_row = module.in_features
                full[key_row, :-1].sub_(full[key_row, -1].clone())
                flat_weight = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat_weight[: module.pruned_index],
                            flat_weight[module.pruned_index + 1 :],
                        )
                    )
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, MeanFreeResidualLinear):
>>>>>>> REPLACE