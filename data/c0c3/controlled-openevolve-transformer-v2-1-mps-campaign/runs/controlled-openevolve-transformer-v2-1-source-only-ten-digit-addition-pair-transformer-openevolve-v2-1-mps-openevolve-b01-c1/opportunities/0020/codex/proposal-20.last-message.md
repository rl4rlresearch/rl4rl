MECHANISM: Multi-query attention with a shared learned key/value memory

HYPOTHESIS: Replacing two independent key/value heads with one shared learned key/value head will reduce parameters from 1,593 to 1,523 while retaining at least 99% accuracy, because the two distinct query heads can retrieve both operand digits from the same content-addressed memory.

INTENDED_EDIT: Keep two learned query heads but broadcast one four-dimensional learned key/value head across them, preserving the original full-width initialization draws so the experiment isolates head sharing.

EVIDENCE: The current two-head model reaches 99.88%, while recent one-parameter affine pruning has become optimization-sensitive and yielded diminishing reductions. This challenges the load-bearing assumption that each query head requires its own key and value projections, while retaining the two independent attention distributions likely needed for simultaneous operand retrieval.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
=======
class MultiQueryQKVLinear(nn.Linear):
    """Independent query heads sharing one learned key/value memory."""

    def __init__(self, d_model: int, n_head: int):
        head_dim = d_model // n_head
        super().__init__(d_model, 3 * d_model)

        retained_rows = torch.cat(
            (
                torch.arange(d_model),
                torch.arange(d_model, d_model + head_dim),
                torch.arange(2 * d_model, 2 * d_model + head_dim),
            )
        )
        full_weight = self.weight.detach()
        full_bias = self.bias.detach()
        self.weight = nn.Parameter(full_weight[retained_rows].clone())
        self.bias = nn.Parameter(full_bias[retained_rows].clone())
        self.full_out_features = 3 * d_model
        self.out_features = d_model + 2 * head_dim
        self.register_buffer("retained_rows", retained_rows, persistent=False)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then prune one key- and one value-bias coordinate.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[: 2 * d_model - 1], full_bias[2 * d_model : -1])).clone()
        )
        self.proj = MeanFreeResidualLinear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = MultiQueryQKVLinear(d_model, n_head)
        self.proj = MeanFreeResidualLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias[:d_model]
        kv_bias = self.qkv.bias[d_model:].view(2, d_model - 1)
        full_bias = torch.cat((q_bias, F.pad(kv_bias, (0, 1)).flatten()))
        qkv = F.linear(x, self.qkv.weight) + full_bias
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.split(
            qkv, (d_model, self.head_dim, self.head_dim), dim=-1
        )

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, 1, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, 1, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, MeanFreePositionEmbedding):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, MultiQueryQKVLinear):
            # Draw the original two-head QKV matrix, then retain both queries
            # and the first head's key/value projections as the shared memory.
            with torch.no_grad():
                full = torch.empty(
                    module.full_out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(full[module.retained_rows])
                nn.init.zeros_(module.bias)
        elif isinstance(module, MeanFreePositionEmbedding):
>>>>>>> REPLACE