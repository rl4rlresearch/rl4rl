MECHANISM: Heterogeneous attention-head bottleneck

HYPOTHESIS: Preserving two independently learned attention heads while narrowing the second head from four to three query/key/value channels will retain at least 99% accuracy with 1,491 parameters, because head-specific addressing—not equal per-head width—appears load-bearing.

INTENDED_EDIT: Replace equal-width 4+4 attention with learned 4+3 heads, removing the final query, key, and value channel of the second head and its corresponding projection input while preserving the original dense initialization draws.

EVIDENCE: Sharing key weights collapsed accuracy to 21.68%, so this patch preserves independent addressing in both heads; unlike the failed one-parameter quotient attempts, it tests whether the auxiliary head needs all four internal channels while retaining one full-width head and the verified 1,524-parameter surrounding model.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class HeterogeneousQKVLinear(nn.Linear):
    """Dense QKV initialization with one channel removed from the second head."""

    def __init__(self, d_model: int, attn_dim: int):
        # Preserve the constructor draw of the original 3*d_model projection.
        super().__init__(d_model, 3 * d_model)
        full_weight = self.weight.detach()
        full_bias = self.bias.detach()

        kept_rows = torch.cat(
            tuple(
                torch.arange(offset, offset + attn_dim)
                for offset in (0, d_model, 2 * d_model)
            )
        )
        self.register_buffer("kept_rows", kept_rows, persistent=False)
        self.full_out_features = 3 * d_model
        self.out_features = 3 * attn_dim
        self.weight = nn.Parameter(full_weight[kept_rows].clone())
        self.bias = nn.Parameter(
            torch.cat(
                (
                    full_bias[:attn_dim],
                    full_bias[2 * d_model : 2 * d_model + attn_dim],
                )
            ).clone()
        )


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove only the softmax-null
        # key bias while retaining the successful full value bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
        self.proj = MeanFreeResidualLinear(d_model, d_model)
=======
        if n_head != 2:
            raise ValueError("heterogeneous attention requires exactly two heads")

        self.n_head = n_head
        full_head_dim = d_model // n_head
        self.head_dims = (full_head_dim, full_head_dim - 1)
        self.attn_dim = sum(self.head_dims)
        self.qkv = HeterogeneousQKVLinear(d_model, self.attn_dim)

        # Construct the original full-width projection to preserve RNG
        # consumption, then remove the channel omitted by the narrow head.
        self.proj = MeanFreeResidualLinear(d_model, d_model)
        full_proj_weight = self.proj.weight.detach()
        self.proj.weight = nn.Parameter(
            full_proj_weight[:, : self.attn_dim].clone()
        )
        self.proj.in_features = self.attn_dim
        self.proj.initialization_in_features = d_model
        self.proj.register_buffer(
            "initialization_input_indices",
            torch.arange(self.attn_dim),
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
        qkv = F.linear(x, self.qkv.weight) + full_bias
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
=======
        _, seqlen, _ = x.shape
        q_bias = self.qkv.bias[: self.attn_dim]
        v_bias = self.qkv.bias[self.attn_dim :]
        full_bias = torch.cat(
            (q_bias, q_bias.new_zeros(self.attn_dim), v_bias)
        )
        qkv = F.linear(x, self.qkv.weight) + full_bias
        q, k, v = qkv.chunk(3, dim=-1)

        q_parts = tuple(
            part.unsqueeze(1) for part in torch.split(q, self.head_dims, dim=-1)
        )
        k_parts = tuple(
            part.unsqueeze(1) for part in torch.split(k, self.head_dims, dim=-1)
        )
        v_parts = tuple(
            part.unsqueeze(1) for part in torch.split(v, self.head_dims, dim=-1)
        )

        att = torch.cat(
            tuple(
                (q_part @ k_part.transpose(-2, -1)) / math.sqrt(head_dim)
                for q_part, k_part, head_dim in zip(
                    q_parts, k_parts, self.head_dims
                )
            ),
            dim=1,
        )
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.cat(
            tuple(
                (att[:, head : head + 1] @ v_part).squeeze(1)
                for head, v_part in enumerate(v_parts)
            ),
            dim=-1,
        )
        y = self.proj(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, DistributedTwelvePrunedInputWeightLinear):
=======
        elif isinstance(module, HeterogeneousQKVLinear):
            # Draw the original dense QKV matrix, retaining the same RNG
            # sequence and the selected channels from each projection.
            with torch.no_grad():
                full = torch.empty(
                    module.full_out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(full[module.kept_rows])
                nn.init.zeros_(module.bias)
        elif isinstance(module, DistributedTwelvePrunedInputWeightLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(module.bias_basis.transpose(0, 1) @ full)
                nn.init.zeros_(module.bias)
=======
                initialization_in_features = getattr(
                    module, "initialization_in_features", module.in_features
                )
                full = torch.empty(
                    module.out_features,
                    initialization_in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                projected = module.bias_basis.transpose(0, 1) @ full
                if hasattr(module, "initialization_input_indices"):
                    projected = projected[
                        :, module.initialization_input_indices
                    ]
                module.weight.copy_(projected)
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE