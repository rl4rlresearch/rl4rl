MECHANISM: Incremental post-initialization key-bias gauge fixing

HYPOTHESIS: Omitting three key-bias coordinates after preserving the baseline initialization sequence will reduce the model from 1,644 to 1,641 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Add a compact QKV projection that reconstructs three omitted key-bias coordinates as zeros, installed only after full baseline initialization.

EVIDENCE: Post-initialization omission of two key-bias coordinates achieved 99.98% accuracy at 1,642 parameters, while omission of all eight failed; removing one additional coordinate is the smallest informative next reduction.

<<<<<<< SEARCH
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
=======
        y = self.resid_drop(y)
        return y


class CompactQKV(nn.Module):
    """QKV projection with three softmax-invariant key-bias coordinates omitted."""

    def __init__(self, linear: nn.Linear, omitted_bias: int):
        super().__init__()
        self.omitted_bias = omitted_bias
        self.weight = linear.weight
        compact_bias = torch.cat(
            (linear.bias[:omitted_bias], linear.bias[omitted_bias + 3 :])
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias[: self.omitted_bias],
                self.bias.new_zeros(3),
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
        # removing three functionally redundant key-bias coordinates.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(block.attn.qkv, cfg.d_model)

    @staticmethod
>>>>>>> REPLACE