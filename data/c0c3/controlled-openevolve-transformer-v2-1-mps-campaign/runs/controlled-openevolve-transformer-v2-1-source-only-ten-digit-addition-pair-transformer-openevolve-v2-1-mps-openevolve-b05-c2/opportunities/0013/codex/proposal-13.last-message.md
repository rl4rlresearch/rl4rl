MECHANISM: Attention value/output-bias redundancy anchoring

HYPOTHESIS: Restoring the qualified 1582-parameter design’s full value bias while fixing one attention output-projection bias coordinate will retain at least 99% accuracy with 1581 parameters, because the missing mean-zero output bias component can be represented through the learned value bias.

INTENDED_EDIT: Adopt the qualified bias-free-ln2 and globally gauge-fixed tied embedding, restore all eight value-bias coordinates, and remove one coordinate only from the attention projection bias.

EVIDENCE: Reference Design 3 reached 99.99% with 1582 parameters and full value bias; the seven-value-bias combination fell to 98.56%, so this patch preserves that sensitive pathway and instead removes one redundant projection-bias coordinate.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        reduced = F.embedding(idx, self.weight)
        return reduced @ self.basis.transpose(0, 1)


class MeanZeroOutputLinear(nn.Linear):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        reduced = F.embedding(idx, self.weight)
        return reduced @ self.basis.transpose(0, 1)


class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with its global all-ones gauge fixed."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        basis = _mean_zero_basis(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(gauged_weight[-1] @ basis)

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(gauged_weight[-1] @ self.basis)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ self.last_weight
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)


class TiedOutputLinear(nn.Module):
    """Parameter-free output view of a gauge-fixed input embedding."""
    def __init__(self, embedding: GaugeFixedEmbedding):
        super().__init__()
        object.__setattr__(self, "embedding", embedding)

        discarded_weight = torch.empty(
            embedding.num_embeddings, embedding.embedding_dim
        )
        nn.init.kaiming_uniform_(discarded_weight, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class MeanZeroOutputLinear(nn.Linear):
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ self.bias
        return F.linear(x, weight, bias)


class CausalSelfAttention(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ self.bias
        return F.linear(x, weight, bias)


class AnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with one bias coordinate absorbed by value bias."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (0, 1))
        return F.linear(x, weight, bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.qv_bias = nn.Parameter(torch.zeros(2, d_model))
        self.proj = AnchoredMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        value_bias = F.pad(self.v_bias, (0, 1))
        qkv_bias = torch.cat(
            (self.q_bias, self.qkv.weight.new_zeros(d_model), value_bias)
        )
=======
        bsz, seqlen, d_model = x.shape
        qkv_bias = torch.cat(
            (self.qv_bias[0], self.qkv.weight.new_zeros(d_model), self.qv_bias[1])
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
=======
        self.token_emb = GaugeFixedEmbedding(cfg.vocab_size, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        # Weight tying with input embeddings.
        self.lm_head = TiedOutputLinear(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, MeanZeroEmbedding):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedEmbedding):
            full_weight = module.weight_rows.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full(full_weight)
        elif isinstance(module, TiedOutputLinear):
            embedding = module.embedding
            full_weight = embedding.weight_rows.new_empty(
                embedding.num_embeddings, embedding.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            embedding.reset_from_full(full_weight)
        elif isinstance(module, MeanZeroEmbedding):
>>>>>>> REPLACE