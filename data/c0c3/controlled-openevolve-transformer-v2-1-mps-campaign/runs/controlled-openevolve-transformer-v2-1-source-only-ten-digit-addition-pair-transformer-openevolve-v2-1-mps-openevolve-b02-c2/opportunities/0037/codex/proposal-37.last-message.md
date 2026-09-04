MECHANISM: Alternating per-head query-bias compaction

HYPOTHESIS: Extending the qualified 1,626-parameter design by omitting a second head-1 query-bias coordinate will produce a 1,625-parameter model with at least 99% accuracy.

INTENDED_EDIT: Apply the verified one-coordinate positional quotient and 4:3 value/all-key bias layout, while retaining three head-0 and two head-1 query-bias coordinates.

EVIDENCE: Omitting one head-1 query bias achieved 99.43% at 1,627 parameters, and then omitting one head-0 query bias improved accuracy to 99.86% at 1,626; continuing the successful alternating reduction on head 1 is the smallest untested QKV-bias change.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and three value biases per head."""

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
                linear.bias[self.value_start + 3 : self.value_start + self.head_dim],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retained_key_end = self.key_start + 1
        first_value_end = retained_key_end + self.head_dim - 3
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(7),
                self.bias[self.key_start : retained_key_end],
                self.bias.new_zeros(3),
                self.bias[retained_key_end:first_value_end],
                self.bias.new_zeros(3),
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
    """QKV projection omitting 1:2 query, all key, and 4:3 value biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[: self.head_dim - 1],
                linear.bias[self.head_dim : key_start - 2],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_query_end = self.head_dim - 1
        second_query_end = first_query_end + self.head_dim - 2
        full_bias = torch.cat(
            (
                self.bias[:first_query_end],
                self.bias.new_zeros(1),
                self.bias[first_query_end:second_query_end],
                self.bias.new_zeros(2),
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[second_query_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and three value biases from each head.
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

        # Preserve every baseline initialization draw before removing one
        # head-0 and two head-1 query biases, all key biases, and 4:3 values.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
>>>>>>> REPLACE