MECHANISM: Anchored key-projection LayerNorm gauge

HYPOTHESIS: Anchoring the final input coordinate of one key-projection row at zero will reduce the model to 1601 parameters while maintaining at least 99% accuracy, because LayerNorm’s centered activation makes that coordinate redundant up to a position-independent key offset that cancels in attention softmax.

INTENDED_EDIT: Replace the combined QKV weight with a 1601-parameter representation that stores seven coordinates of the first key row, fixes its eighth coordinate to zero, and transforms the original initialization to preserve the initial attention function.

EVIDENCE: Centering the first key row reached 95.84% at 1605 parameters, making this exact key-only gauge a near-threshold candidate; the sharply different outcomes of coordinate-selective value-bias removals further motivate testing an anchored, less gradient-coupled parameterization instead of the failed zero-sum reconstruction.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class AnchoredKeyLinear(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Consume the same constructor RNG as the original QKV linear layer.
        _ = nn.Linear(d_model, 3 * d_model)
        self.d_model = d_model
        self.before_key = nn.Parameter(torch.empty(d_model, d_model))
        self.key_first_row = nn.Parameter(torch.empty(d_model - 1))
        self.after_key = nn.Parameter(torch.empty(2 * d_model - 1, d_model))

    def _set_from_full(self, weight: torch.Tensor) -> None:
        key_row = weight[self.d_model]
        with torch.no_grad():
            self.before_key.copy_(weight[: self.d_model])
            self.key_first_row.copy_(key_row[:-1] - key_row[-1])
            self.after_key.copy_(weight[self.d_model + 1 :])

    def reconstructed_weight(self) -> torch.Tensor:
        key_row = torch.cat(
            (self.key_first_row, self.key_first_row.new_zeros(1))
        )
        return torch.cat(
            (self.before_key, key_row.unsqueeze(0), self.after_key),
            dim=0,
        )


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Construct with the original bias first to preserve the baseline RNG
        # sequence, then retain only the functionally relevant q/v biases.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 6))
        self.qkv.bias = None
=======
        # Preserve the original constructor RNG while anchoring one redundant
        # key-row coordinate outside the learned parameterization.
        self.qkv = AnchoredKeyLinear(d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight, fused_bias)
=======
        qkv = F.linear(x, self.qkv.reconstructed_weight(), fused_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
=======
        elif isinstance(module, AnchoredKeyLinear):
            full_weight = module.before_key.new_empty(
                3 * module.d_model, module.d_model
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
>>>>>>> REPLACE