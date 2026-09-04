MECHANISM: Tied-embedding global translation gauge quotient

HYPOTHESIS: Removing the tied embedding’s globally uniform scalar direction from the verified 1,622-parameter design will yield 1,621 parameters while maintaining at least 99% accuracy, because LayerNorm cancels its input effect and softmax cancels its uniform logit shift.

INTENDED_EDIT: Reproduce the qualified shared-query, compact-ln2, and shared value/projection-bias design, then gauge-fix one globally redundant tied-embedding scalar.

EVIDENCE: The 1,622-parameter shared value/projection design achieved 99.92%; the prior 1,621 attempt instead constrained sensitive ln2 offsets and fell to 62.33%, motivating an untouched exact invariance outside attention and MLP capacity.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection omitting one query bias, all key biases, and 4:3 value biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[: key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query_end = self.key_start - 1
        full_bias = torch.cat(
            (
                self.bias[:query_end],
                self.bias.new_zeros(1),
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[query_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)


class MLP(nn.Module):
=======
class CompactQKV(nn.Module):
    """QKV projection with shared head-0 queries and one retained value bias."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        query_bias = torch.cat(
            (
                linear.bias[:1],
                linear.bias[self.head_dim : key_start - 1],
            )
        )
        self.query_bias = nn.Parameter(query_bias.detach().clone())
        self.value_bias = nn.Parameter(
            linear.bias[self.value_start + self.head_dim + 3 :].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.query_bias[:1].expand(self.head_dim - 2),
                self.query_bias.new_zeros(2),
                self.query_bias[1:],
                self.query_bias.new_zeros(1),
                self.query_bias.new_zeros(self.key_start),
                self.query_bias.new_zeros(self.head_dim),
                self.query_bias.new_zeros(3),
                self.value_bias,
            )
        )
        return F.linear(x, self.weight, full_bias)


class CompactSharedProjection(nn.Module):
    """Attention projection sharing its final bias with the retained value bias."""

    def __init__(self, linear: nn.Linear, shared_bias: nn.Parameter):
        super().__init__()
        self.weight = linear.weight
        self.bias = nn.Parameter(linear.bias[:-1].detach().clone())
        self.shared_bias = shared_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.shared_bias))
        return F.linear(x, self.weight, full_bias)


class CompactLayerNormBias(nn.Module):
    """LayerNorm with one downstream-linear-absorbed bias coordinate fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-1].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            full_bias,
            self.eps,
        )


class CompactTokenEmbedding(nn.Module):
    """Tied embedding with its globally uniform translation gauge fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        flat_weight = embedding.weight.detach().reshape(-1)
        gauge_fixed = flat_weight - flat_weight[0]
        self.weight = nn.Parameter(gauge_fixed[1:].clone())

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight.new_zeros(1), self.weight)).view(
            self.num_embeddings,
            self.embedding_dim,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class CompactTiedHead(nn.Module):
    """Parameter-free output view of the compact learned token embedding."""

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weight)


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = CompactLayerNormBias(nn.LayerNorm(cfg.d_model))
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix one exact token/position translation gauge while preserving the
        # initialized hidden inputs and output probabilities.
        with torch.no_grad():
            offset = self.pos_emb.weight[0, 0].detach().clone()
            self.token_emb.weight[:, 0].add_(offset)
            self.pos_emb.weight[:, 0].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)

        # Preserve every baseline initialization draw before removing one
        # head-1 query bias, all key biases, and value biases in a 4:3 layout.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
        # Fix one exact token/position translation gauge while preserving the
        # initialized hidden inputs and output probabilities.
        with torch.no_grad():
            offset = self.pos_emb.weight[0, 0].detach().clone()
            self.token_emb.weight[:, 0].add_(offset)
            self.pos_emb.weight[:, 0].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)

        # A scalar added to every embedding entry is removed from hidden states
        # by LayerNorm and becomes a vocabulary-uniform output-logit shift.
        self.token_emb = CompactTokenEmbedding(self.token_emb)
        self.lm_head = CompactTiedHead()

        # Retain the qualified QKV layout, then quotient its constant-offset
        # redundancy by sharing the final value and projection bias scalar.
        for block in self.blocks:
            compact_qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
            block.attn.qkv = compact_qkv
            block.attn.proj = CompactSharedProjection(
                block.attn.proj,
                compact_qkv.value_bias,
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = self.lm_head(x, self.token_emb.full_weight())
>>>>>>> REPLACE