MECHANISM: Token-position translation gauge fixing

HYPOTHESIS: Applying an exact token/position embedding translation quotient to the qualified 4:3 value-bias design will produce a 1,629-parameter model with at least 99% accuracy.

INTENDED_EDIT: Adopt the verified 4:3 value-bias omission, transfer one positional scalar into the tied token embeddings, and reconstruct that positional scalar as zero.

EVIDENCE: The 4:3 value-bias layout achieved 99.81% at 1,630 parameters. Prior 1,629 changes to attention-output and `ln2` parameters failed, motivating a distinct embedding gauge that preserves initialized hidden inputs and changes output logits only by a class-independent shift.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and two value biases per head."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[second_head_start + 3 : self.value_start],
                linear.bias[self.value_start + 2 : self.value_start + self.head_dim],
                linear.bias[self.value_start + self.head_dim + 2 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retained_key_end = self.key_start + 1
        first_value_end = retained_key_end + self.head_dim - 2
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(7),
                self.bias[self.key_start : retained_key_end],
                self.bias.new_zeros(2),
                self.bias[retained_key_end:first_value_end],
                self.bias.new_zeros(2),
                self.bias[first_value_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class CompactPositionEmbedding(nn.Module):
    """Positional embedding with one token-position translation gauge fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[1:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(1), self.weight))
        return F.embedding(
            idx,
            full_weight.view(self.num_embeddings, self.embedding_dim),
        )


class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and value biases in a 4:3 layout."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[second_head_start + 3 : self.value_start],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retained_key_end = self.key_start + 1
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(7),
                self.bias[self.key_start : retained_key_end],
                self.bias.new_zeros(4),
                self.bias.new_zeros(3),
                self.bias[retained_key_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and two value biases from each head.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
        self.apply(self._init_weights)

        # Fix one exact token/position translation gauge while preserving the
        # initialized hidden inputs and output probabilities.
        with torch.no_grad():
            offset = self.pos_emb.weight[0, 0].detach().clone()
            self.token_emb.weight[:, 0].add_(offset)
            self.pos_emb.weight[:, 0].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)

        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and value biases in a 4:3 per-head layout.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
>>>>>>> REPLACE