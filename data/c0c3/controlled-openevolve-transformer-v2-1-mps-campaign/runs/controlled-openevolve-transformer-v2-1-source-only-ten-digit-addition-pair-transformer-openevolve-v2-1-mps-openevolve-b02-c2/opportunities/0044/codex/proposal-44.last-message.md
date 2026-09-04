MECHANISM: Final-coordinate LayerNorm–MLP scaling gauge quotient

HYPOTHESIS: Fixing one `ln2` scale coordinate in the qualified 1,624-parameter design will yield 1,623 parameters and at least 99% accuracy because the downstream `fc1` column retains the removed scaling degree of freedom.

INTENDED_EDIT: Reproduce the verified positional and QKV compactions, then reconstruct the final `ln2` scale as one while learning its other seven scales and all biases.

EVIDENCE: The 1,624-parameter shared head-0 query-bias design achieved 99.64%; prior 1,623 attempts modified fragile attention coordinates, whereas this reduction preserves that qualified attention layout and removes an exact scale redundancy immediately before a free MLP input column.

<<<<<<< SEARCH
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
=======
        y = self.resid_drop(y)
        return y


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
    """QKV projection sharing 2 head-0 query biases and retaining 3 head-1 biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:1],
                linear.bias[self.head_dim : key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        second_query_end = 1 + self.head_dim - 1
        full_bias = torch.cat(
            (
                self.bias[:1].expand(self.head_dim - 2),
                self.bias.new_zeros(2),
                self.bias[1:second_query_end],
                self.bias.new_zeros(1),
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[second_query_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)


class CompactLayerNorm(nn.Module):
    """LayerNorm fixing one scale absorbed by the following linear input column."""

    def __init__(self, layer: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer.normalized_shape
        self.eps = layer.eps
        self.weight = nn.Parameter(layer.weight.detach()[:-1].clone())
        self.bias = layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(
            x,
            self.normalized_shape,
            full_weight,
            self.bias,
            self.eps,
        )


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # Fix one exact token/position translation gauge while preserving the
        # initialized hidden inputs and output probabilities.
        with torch.no_grad():
            offset = self.pos_emb.weight[0, 0].detach().clone()
            self.token_emb.weight[:, 0].add_(offset)
            self.pos_emb.weight[:, 0].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)

        # Preserve the qualified shared 2:3 query and 0:1 value layouts with
        # no key biases, then quotient one ln2-to-fc1 scaling redundancy.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
            block.ln2 = CompactLayerNorm(block.ln2)

    @staticmethod
>>>>>>> REPLACE