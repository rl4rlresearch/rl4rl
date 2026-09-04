MECHANISM: Global tied-embedding shift quotient

HYPOTHESIS: Restoring the successful full value bias while removing one exact global-shift degree of freedom from the tied token/output embedding will yield 1,585 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Restore all eight value-bias coordinates, then represent the tied embedding in an orthonormal 111-dimensional mean-free basis while preserving constructor and initialization RNG consumption.

EVIDENCE: The 1,586-parameter key-bias-free design achieved 99.94%, whereas pruning one value-bias coordinate fell to 97.31%. Mean-free quotient parameterizations already preserved accuracy for positional embeddings and residual outputs, motivating an exact tied-embedding gauge reduction instead.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        coordinates = F.embedding(idx, self.weight)
        return coordinates @ self.basis.transpose(0, 1)


class CausalSelfAttention(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        coordinates = F.embedding(idx, self.weight)
        return coordinates @ self.basis.transpose(0, 1)


class GaugeFixedTiedEmbedding(nn.Module):
    """Tied embedding modulo its globally constant, function-invariant shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        flat_dim = num_embeddings * embedding_dim

        # Preserve the constructor RNG consumption of nn.Embedding.
        constructor_probe = nn.Embedding(num_embeddings, embedding_dim)
        full_weight = constructor_probe.weight.detach().reshape(-1)

        basis = torch.zeros(flat_dim, flat_dim - 1)
        for j in range(flat_dim - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
        self.weight = nn.Parameter((full_weight @ basis).clone())

    def full_weight(self) -> torch.Tensor:
        flat_weight = self.basis @ self.weight
        return flat_weight.view(self.num_embeddings, self.embedding_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class TiedOutputHead(nn.Module):
    """Output projection sharing the gauge-fixed token embedding."""

    def __init__(self, embedding: GaugeFixedTiedEmbedding):
        super().__init__()
        # Preserve the constructor RNG consumption of the original nn.Linear.
        constructor_probe = nn.Linear(
            embedding.embedding_dim, embedding.num_embeddings, bias=False
        )
        del constructor_probe
        self.embedding = embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding.full_weight())


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove the softmax-null key bias
        # and prune one value-bias coordinate.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model : -1])).clone()
        )
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove the softmax-null key bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), F.pad(v_bias, (0, 1)))
        )
=======
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanFreePositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        self.cfg = cfg
        self.token_emb = GaugeFixedTiedEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanFreePositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = TiedOutputHead(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, MeanFreePositionEmbedding):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedTiedEmbedding):
            # Draw the original full embedding initialization, then remove only
            # its globally constant component.
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings * module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(full @ module.basis)
        elif isinstance(module, MeanFreePositionEmbedding):
>>>>>>> REPLACE