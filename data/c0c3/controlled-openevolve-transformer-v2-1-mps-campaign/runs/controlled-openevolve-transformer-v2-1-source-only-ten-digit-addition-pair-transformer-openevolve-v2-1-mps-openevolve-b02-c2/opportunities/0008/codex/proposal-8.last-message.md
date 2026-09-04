MECHANISM: Incremental post-initialization key-bias gauge fixing

HYPOTHESIS: Omitting two key-bias coordinates after preserving the baseline initialization sequence will reduce the model to 1,642 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Replace each initialized QKV projection with an equivalent module that retains 22 learned bias values and reconstructs the first two softmax-invariant key-bias coordinates as zeros.

EVIDENCE: Post-initialization omission of one key-bias coordinate achieved 99.96% at 1,643 parameters, while omitting all eight collapsed to 30.16%; removing exactly one additional coordinate is the smallest informative capacity reduction.

<<<<<<< SEARCH
class MLP(nn.Module):
=======
class CompactQKV(nn.Module):
    """QKV projection with two softmax-invariant key-bias coordinates omitted."""

    def __init__(self, linear: nn.Linear, omitted_bias: int):
        super().__init__()
        self.omitted_bias = omitted_bias
        self.weight = linear.weight
        compact_bias = torch.cat(
            (linear.bias[:omitted_bias], linear.bias[omitted_bias + 2 :])
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias[: self.omitted_bias],
                self.bias.new_zeros(2),
                self.bias[self.omitted_bias :],
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

        # Preserve the qualified baseline initialization trajectory before
        # removing two functionally redundant key-bias coordinates.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(block.attn.qkv, cfg.d_model)

    @staticmethod
>>>>>>> REPLACE