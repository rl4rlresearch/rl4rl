MECHANISM: Shared-key multi-head causal attention

HYPOTHESIS: Sharing one learned four-dimensional key encoder across both attention heads will reduce the model from 1,533 to 1,501 parameters while retaining at least 99% accuracy, because head-specific queries and values can still retrieve different operands through a common learned positional-address space.

INTENDED_EDIT: Replace the independent per-head key projections with one learned key projection broadcast across both heads, while retaining separate query and value projections and preserving the original full-QKV initialization draw and variance.

EVIDENCE: The 1,533-parameter design succeeds, but repeated isolated gauge removals are optimization-brittle, and rank-five token compression lost essential token geometry. This challenges the shared assumption that each head needs an independent key dictionary: fixed-format addition plausibly needs a common address representation, while distinct queries and values preserve head specialization.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

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
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        y = self.resid_drop(y)
        return y
=======
class SharedKeyQKV(nn.Module):
    """Separate queries and values with one learned key dictionary for all heads."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        self.head_dim = d_model // n_head

        # Consume the same constructor draw as the original full QKV layer.
        original = nn.Linear(d_model, 3 * d_model)
        full_weight = original.weight.detach()
        full_bias = original.bias.detach()
        key_start = d_model
        key_middle = d_model + self.head_dim
        key_end = 2 * d_model
        shared_key = (
            full_weight[key_start:key_middle]
            + full_weight[key_middle:key_end]
        ) / math.sqrt(2.0)

        self.q_weight = nn.Parameter(full_weight[:d_model].clone())
        self.key_weight = nn.Parameter(shared_key.clone())
        self.v_weight = nn.Parameter(full_weight[2 * d_model :].clone())
        self.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_bias = self.bias[: self.d_model]
        v_bias = self.bias[self.d_model :]
        q = F.linear(x, self.q_weight, q_bias)
        k = F.linear(x, self.key_weight)
        v = F.linear(x, self.v_weight, v_bias)
        return q, k, v


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = SharedKeyQKV(d_model, n_head)
        self.proj = MeanFreeResidualLinear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q, k, v = self.qkv(x)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, 1, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
        return y
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanFreeTokenEmbedding):
            with torch.no_grad():
=======
        elif isinstance(module, SharedKeyQKV):
            # Preserve the original full-QKV RNG draw. Averaging the two
            # independent key blocks with variance correction initializes the
            # shared dictionary at the same per-coordinate scale.
            with torch.no_grad():
                full = torch.empty(
                    3 * module.d_model,
                    module.d_model,
                    device=module.q_weight.device,
                    dtype=module.q_weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                key_start = module.d_model
                key_middle = module.d_model + module.head_dim
                key_end = 2 * module.d_model
                module.q_weight.copy_(full[: module.d_model])
                module.key_weight.copy_(
                    (
                        full[key_start:key_middle]
                        + full[key_middle:key_end]
                    )
                    / math.sqrt(2.0)
                )
                module.v_weight.copy_(full[2 * module.d_model :])
                nn.init.zeros_(module.bias)
        elif isinstance(module, MeanFreeTokenEmbedding):
            with torch.no_grad():
>>>>>>> REPLACE