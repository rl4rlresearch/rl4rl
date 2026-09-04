MECHANISM: Post-initialization redundant key-bias gauge fixing

HYPOTHESIS: Removing one key-bias coordinate only after the complete 1,644-parameter model has consumed its original construction and initialization RNG sequence will produce a 1,643-parameter model with at least 99% accuracy.

INTENDED_EDIT: Preserve the successful model’s full initialization sequence, then replace each QKV projection with an equivalent projection that reconstructs one omitted key-bias coordinate as zero.

EVIDENCE: The baseline reached 99.96%, while prior mathematically redundant bias reductions failed despite preserving selected weights; this isolates whether perturbing the full initialization RNG trajectory caused those optimization failures.

<<<<<<< SEARCH
class MLP(nn.Module):
=======
class CompactQKV(nn.Module):
    """QKV projection with one softmax-invariant key-bias coordinate omitted."""

    def __init__(self, linear: nn.Linear, omitted_bias: int):
        super().__init__()
        self.omitted_bias = omitted_bias
        self.weight = linear.weight
        compact_bias = torch.cat(
            (linear.bias[:omitted_bias], linear.bias[omitted_bias + 1 :])
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias[: self.omitted_bias],
                self.bias.new_zeros(1),
                self.bias[self.omitted_bias :],
            )
        )
        return F.linear(x, self.weight, full_bias)


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
=======
        self.apply(self._init_weights)

        # Compact only after every module has consumed the baseline initialization
        # draws, so all retained parameters start identically to the qualified run.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(block.attn.qkv, cfg.d_model)
>>>>>>> REPLACE