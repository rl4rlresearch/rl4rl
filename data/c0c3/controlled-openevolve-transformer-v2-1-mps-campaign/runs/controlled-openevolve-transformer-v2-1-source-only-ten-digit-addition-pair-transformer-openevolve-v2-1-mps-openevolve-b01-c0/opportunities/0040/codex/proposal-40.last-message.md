MECHANISM: Softmax-null centered key-projection row

HYPOTHESIS: Centering one key-projection weight row will reduce the verified model from 1619 to 1618 parameters while retaining at least 99% accuracy, because the eliminated component produces only a position-independent key offset on the LayerNorm affine hyperplane and therefore cannot affect causal attention probabilities.

INTENDED_EDIT: Replace the first eight-coordinate key-weight row with seven learned centered contrasts, retain every other QKV weight unchanged, and reconstruct the centered original initialization without altering the RNG sequence.

EVIDENCE: The verified 1619-parameter design already removes every key bias while achieving 99.92%, demonstrating that position-independent key offsets are safely softmax-null; this targets the corresponding key-weight redundancy instead of extending value-bias, output-column, or LayerNorm-scale changes that recently collapsed.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class ReducedKeyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Consume the same constructor RNG as the original linear layer.
        _ = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.first_key_row = nn.Parameter(torch.empty(in_features - 1))
        self.other_rows = nn.Parameter(torch.empty(out_features - 1, in_features))

    def _set_from_full(self, weight: torch.Tensor) -> None:
        key_index = self.in_features
        key_row = weight[key_index]
        with torch.no_grad():
            self.first_key_row.copy_(key_row[:-1] - key_row[-1])
            self.other_rows.copy_(
                torch.cat((weight[:key_index], weight[key_index + 1 :]), dim=0)
            )

    def full_weight(self) -> torch.Tensor:
        anchored = torch.cat(
            (self.first_key_row, self.first_key_row.new_zeros(1))
        )
        key_row = anchored - anchored.mean()
        key_index = self.in_features
        return torch.cat(
            (
                self.other_rows[:key_index],
                key_row.unsqueeze(0),
                self.other_rows[key_index:],
            ),
            dim=0,
        )


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 1))
        self.qkv.bias = None
=======
        self.qkv = ReducedKeyLinear(d_model, 3 * d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight, fused_bias)
=======
        qkv = F.linear(x, self.qkv.full_weight(), fused_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, nn.Linear):
            if module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
=======
        if isinstance(module, nn.Linear):
            if module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, ReducedKeyLinear):
            full_weight = module.first_key_row.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, nn.Embedding):
>>>>>>> REPLACE