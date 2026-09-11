MECHANISM: Single-row key-projection gauge removal

HYPOTHESIS: Centering one key-projection row will reduce the model to 1605 parameters while maintaining at least 99% accuracy, because the removed LayerNorm-input gauge component produces only a position-independent key offset that cancels in the attention softmax.

INTENDED_EDIT: Replace the combined QKV weight with equivalent storage that reconstructs the first key row from seven learned coordinates with zero sum, while preserving the original initialization draw and initial model function.

EVIDENCE: The 1606-parameter design reached 99.89%, while sharing sensitive value-bias coordinates and constraining attention-output columns failed; this motivates reducing a key-only degree of freedom whose effect cancels directly inside attention rather than constraining the residual output path.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class CenteredKeyQKVLinear(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Consume the same constructor RNG as the original QKV linear layer.
        _ = nn.Linear(d_model, 3 * d_model)
        self.d_model = d_model
        self.q_rows = nn.Parameter(torch.empty(d_model, d_model))
        self.first_key_row = nn.Parameter(torch.empty(d_model - 1))
        self.remaining_rows = nn.Parameter(
            torch.empty(2 * d_model - 1, d_model)
        )

    def _set_from_full(self, weight: torch.Tensor) -> None:
        first_key = weight[self.d_model]
        centered_first_key = first_key - first_key.mean()
        with torch.no_grad():
            self.q_rows.copy_(weight[: self.d_model])
            self.first_key_row.copy_(centered_first_key[:-1])
            self.remaining_rows.copy_(weight[self.d_model + 1 :])

    def tied_weight(self) -> torch.Tensor:
        final_coordinate = -self.first_key_row.sum().reshape(1)
        first_key = torch.cat((self.first_key_row, final_coordinate))
        return torch.cat(
            (
                self.q_rows,
                first_key.unsqueeze(0),
                self.remaining_rows,
            ),
            dim=0,
        )


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 2))
        self.qkv.bias = None
=======
        self.qkv = CenteredKeyQKVLinear(d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight, fused_bias)
=======
        qkv = F.linear(x, self.qkv.tied_weight(), fused_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, CenteredTokenEmbedding):
=======
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, CenteredKeyQKVLinear):
            full_weight = module.q_rows.new_empty(
                3 * module.d_model, module.d_model
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, CenteredTokenEmbedding):
>>>>>>> REPLACE