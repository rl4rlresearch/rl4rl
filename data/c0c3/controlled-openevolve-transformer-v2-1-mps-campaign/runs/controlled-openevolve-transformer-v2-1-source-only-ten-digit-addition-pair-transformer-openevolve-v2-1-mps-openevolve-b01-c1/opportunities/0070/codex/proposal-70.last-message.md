MECHANISM: Query-routed shared attention memory

HYPOTHESIS: Replacing head-specific key/value projections with one learned shared key/value projection will reduce the model from 1,549 to 1,481 parameters while retaining at least 99% accuracy, because independent learned queries still let the two heads retrieve different positions while the retrieved operand-pair representation can be shared.

INTENDED_EDIT: Convert causal self-attention to multi-query attention: retain two independent query heads, share one learned four-dimensional key/value memory across them, and preserve the original initialization distribution and RNG sequence.

EVIDENCE: The 1,549-parameter model achieves 99.98% accuracy, while failures from pruning positional or normalization coordinates show that representation channels are load-bearing. This instead challenges the untested assumption that both heads need separate key/value feature maps; distinct query projections and output slots remain intact.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class MultiQueryQKV(nn.Linear):
    """Independent query heads addressing one shared learned key/value memory."""

    def __init__(self, d_model: int, n_head: int):
        # Preserve the constructor RNG consumption of the original 3*d_model
        # projection before retaining the shared-head subspace.
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        full_weight = self.weight.detach()
        full_bias = self.bias.detach()
        shared_key = (
            full_weight[d_model : 2 * d_model]
            .reshape(n_head, self.head_dim, d_model)
            .sum(dim=0)
            / math.sqrt(n_head)
        )
        shared_value = (
            full_weight[2 * d_model :]
            .reshape(n_head, self.head_dim, d_model)
            .sum(dim=0)
            / math.sqrt(n_head)
        )
        shared_value_bias = (
            full_bias[2 * d_model :]
            .reshape(n_head, self.head_dim)
            .sum(dim=0)
            / math.sqrt(n_head)
        )

        self.weight = nn.Parameter(
            torch.cat((full_weight[:d_model], shared_key, shared_value)).clone()
        )
        self.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], shared_value_bias)).clone()
        )
        self.out_features = d_model + 2 * self.head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_bias = self.bias[: self.d_model]
        v_bias = self.bias[self.d_model :]
        full_bias = torch.cat(
            (q_bias, q_bias.new_zeros(self.head_dim), v_bias)
        )
        return F.linear(x, self.weight, full_bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove only the softmax-null
        # key bias while retaining the successful full value bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
=======
        self.qkv = MultiQueryQKV(d_model, n_head)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
        qkv = F.linear(x, self.qkv.weight) + full_bias
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = torch.split(
            qkv, (d_model, self.head_dim, self.head_dim), dim=-1
        )

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, 1, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, 1, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanFreeResidualLinear):
=======
        elif isinstance(module, MultiQueryQKV):
            # Initialize from the equal-head subspace of the original full QKV
            # draw, preserving its variance and all subsequent RNG state.
            with torch.no_grad():
                full = torch.empty(
                    3 * module.d_model,
                    module.d_model,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shared_key = (
                    full[module.d_model : 2 * module.d_model]
                    .reshape(
                        module.n_head,
                        module.head_dim,
                        module.d_model,
                    )
                    .sum(dim=0)
                    / math.sqrt(module.n_head)
                )
                shared_value = (
                    full[2 * module.d_model :]
                    .reshape(
                        module.n_head,
                        module.head_dim,
                        module.d_model,
                    )
                    .sum(dim=0)
                    / math.sqrt(module.n_head)
                )
                module.weight.copy_(
                    torch.cat((full[: module.d_model], shared_key, shared_value))
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, MeanFreeResidualLinear):
>>>>>>> REPLACE