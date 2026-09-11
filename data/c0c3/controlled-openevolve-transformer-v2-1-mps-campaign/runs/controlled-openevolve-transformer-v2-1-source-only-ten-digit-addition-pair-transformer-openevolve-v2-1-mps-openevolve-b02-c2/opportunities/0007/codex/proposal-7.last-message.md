MECHANISM: Post-initialization removal of the complete softmax-invariant key bias

HYPOTHESIS: Compacting all eight key-bias coordinates only after baseline initialization will produce a 1,636-parameter model with at least 99% accuracy.

INTENDED_EDIT: Preserve the successful model’s complete initialization sequence, then replace each QKV projection with one retaining learned query/value biases while reconstructing the redundant key bias as zeros.

EVIDENCE: Post-initialization removal of one key-bias coordinate achieved 99.96% at 1,643 parameters, while the earlier 1,636-parameter attempt changed construction before initialization; this motivates testing the full exact invariance with the now-qualified RNG-preserving method.

<<<<<<< SEARCH
class MLP(nn.Module):
=======
class CompactQKV(nn.Module):
    """QKV projection with the softmax-invariant key bias omitted."""

    def __init__(self, linear: nn.Linear, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.weight = linear.weight
        compact_bias = torch.cat(
            (linear.bias[:d_model], linear.bias[2 * d_model :])
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias[: self.d_model],
                self.bias.new_zeros(self.d_model),
                self.bias[self.d_model :],
            )
        )
        return F.linear(x, self.weight, full_bias)


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # Compact only after every module has consumed the baseline initialization
        # draws, so all retained parameters start identically to the qualified run.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(block.attn.qkv, cfg.d_model)

    @staticmethod
>>>>>>> REPLACE