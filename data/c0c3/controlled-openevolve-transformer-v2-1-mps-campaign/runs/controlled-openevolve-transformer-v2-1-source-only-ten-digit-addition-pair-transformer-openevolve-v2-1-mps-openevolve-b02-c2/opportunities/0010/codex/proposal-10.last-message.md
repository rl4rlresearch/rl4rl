MECHANISM: Head-balanced post-initialization key-bias gauge fixing

HYPOTHESIS: Omitting three redundant key-bias coordinates distributed 2:1 across the two attention heads will produce a 1,641-parameter model with at least 99% accuracy.

INTENDED_EDIT: Preserve the baseline initialization sequence, then replace QKV with a compact projection omitting the first two key-bias coordinates of head 0 and the first key-bias coordinate of head 1.

EVIDENCE: Omitting two key-bias coordinates achieved 99.98% at 1,642 parameters, while omitting three consecutive coordinates failed at 71.13%; distributing the third omission across heads tests whether the failure arose from concentrating all three omissions in one head.

<<<<<<< SEARCH
class MLP(nn.Module):
=======
class CompactQKV(nn.Module):
    """QKV projection with three head-balanced key-bias coordinates omitted."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[key_start + 2 : second_head_start],
                linear.bias[second_head_start + 1 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        middle_start = self.key_start
        middle_end = self.second_head_start - 2
        full_bias = torch.cat(
            (
                self.bias[:middle_start],
                self.bias.new_zeros(2),
                self.bias[middle_start:middle_end],
                self.bias.new_zeros(1),
                self.bias[middle_end:],
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

        # Preserve every baseline initialization draw before removing redundant
        # key-bias coordinates, distributing the omissions across both heads.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )

    @staticmethod
>>>>>>> REPLACE