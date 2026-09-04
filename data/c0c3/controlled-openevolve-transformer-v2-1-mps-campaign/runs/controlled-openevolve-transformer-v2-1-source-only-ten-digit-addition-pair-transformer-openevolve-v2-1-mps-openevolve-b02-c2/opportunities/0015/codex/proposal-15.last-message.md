MECHANISM: Post-initialization positional row-shift gauge fixing

HYPOTHESIS: Retaining the qualified 4:3 key-bias layout while fixing one redundant positional-embedding coordinate will produce a 1,636-parameter model with at least 99% accuracy.

INTENDED_EDIT: Upgrade QKV compaction from 3:2 to the qualified 4:3 layout, then preserve the initialized positional embedding exactly while making its first coordinate non-trainable; per-position uniform hidden-state shifts are removed by every pre-LayerNorm path and the final LayerNorm.

EVIDENCE: The 4:3 QKV design achieved 99.89% at 1,637 parameters, while 4:4 failed at 30.16%; retaining the remaining key-bias coordinate and removing an independent one-dimensional positional gauge is the smallest informative reduction.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection with key-bias omissions distributed 3:2 across heads."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[key_start + 3 : second_head_start],
                linear.bias[second_head_start + 2 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        middle_start = self.key_start
        middle_end = self.second_head_start - 3
        full_bias = torch.cat(
            (
                self.bias[:middle_start],
                self.bias.new_zeros(3),
                self.bias[middle_start:middle_end],
                self.bias.new_zeros(2),
                self.bias[middle_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class CompactQKV(nn.Module):
    """QKV projection with key-bias omissions distributed 4:3 across heads."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[key_start + 4 : second_head_start],
                linear.bias[second_head_start + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        middle_start = self.key_start
        middle_end = self.second_head_start - 4
        full_bias = torch.cat(
            (
                self.bias[:middle_start],
                self.bias.new_zeros(4),
                self.bias[middle_start:middle_end],
                self.bias.new_zeros(3),
                self.bias[middle_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)


class CompactPosEmbedding(nn.Module):
    """Position embedding with one fixed row-shift gauge coordinate."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        flat_weight = embedding.weight.detach().reshape(-1)
        self.register_buffer("fixed", flat_weight[:1].clone())
        self.weight = nn.Parameter(flat_weight[1:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.fixed, self.weight)).view(
            self.num_embeddings, self.embedding_dim
        )
        return F.embedding(idx, full_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing five
        # redundant key-bias coordinates, distributed 3:2 across the heads.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
        # Preserve every baseline initialization draw before removing seven
        # redundant key-bias coordinates, distributed 4:3 across the heads.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )

        # Preserve the initialized function while fixing one coordinate along
        # the position-specific uniform-shift gauge removed by LayerNorm.
        self.pos_emb = CompactPosEmbedding(self.pos_emb)
>>>>>>> REPLACE