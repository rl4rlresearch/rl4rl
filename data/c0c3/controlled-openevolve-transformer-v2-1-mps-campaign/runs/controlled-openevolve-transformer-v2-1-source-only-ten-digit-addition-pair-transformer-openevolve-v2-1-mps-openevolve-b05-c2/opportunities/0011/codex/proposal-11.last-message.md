MECHANISM: Vocabulary-common-mode gauge compression

HYPOTHESIS: Centering the tied token/output embeddings across vocabulary will reduce the qualified 1583-parameter design to 1575 parameters while retaining at least 99% accuracy, because the removed vector produces only a common logit shift and its input-side effect is absorbable by positional embeddings up to LayerNorm-invisible common mode.

INTENDED_EDIT: Apply the qualified mean-zero residual parameterization and bias-free `ln2`, then represent tied token/output embeddings in an orthonormal vocabulary-centered basis while preserving the qualified model’s initial function.

EVIDENCE: The 1583-parameter mean-zero, bias-free-`ln2` design achieved 99.96%; prior smaller failures altered sensitive value-bias or LayerNorm-scale paths, whereas this patch removes a distinct exact tied-embedding/softmax gauge without reducing attention, MLP, or normalization capacity.

<<<<<<< SEARCH
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


def _mean_zero_basis(dim: int) -> torch.Tensor:
    """Orthonormal basis for vectors whose coordinates sum to zero."""
    if dim < 2:
        raise ValueError("mean-zero features require dimension >= 2")
    basis = torch.zeros(dim, dim - 1)
    for col in range(dim - 1):
        scale = 1.0 / math.sqrt((col + 1) * (col + 2))
        basis[: col + 1, col] = scale
        basis[col + 1, col] = -(col + 1) * scale
    return basis


class VocabCenteredEmbedding(nn.Module):
    """Tied embeddings modulo the softmax-invisible common vocabulary row."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        basis = _mean_zero_basis(num_embeddings)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        self.weight = nn.Parameter(basis.transpose(0, 1) @ full_weight)

    def full_weight(self) -> torch.Tensor:
        return self.basis @ self.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class VocabCenteredLMHead(nn.Linear):
    """Output head sharing the vocabulary-centered embedding coordinates."""
    def __init__(self, embedding: VocabCenteredEmbedding):
        super().__init__(
            embedding.embedding_dim, embedding.num_embeddings, bias=False
        )
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = embedding.weight
        self.register_buffer("basis", embedding.basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.basis @ self.weight)


class MeanZeroEmbedding(nn.Module):
    """Learn an embedding modulo its LayerNorm-invisible common mode."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        basis = _mean_zero_basis(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        self.weight = nn.Parameter(full_weight @ basis)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        reduced = F.embedding(idx, self.weight)
        return reduced @ self.basis.transpose(0, 1)


class MeanZeroOutputLinear(nn.Linear):
    """Linear map with its residual-stream common mode gauge-fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        basis = _mean_zero_basis(out_features)
        self.weight = nn.Parameter(basis.transpose(0, 1) @ self.weight.detach())
        self.bias = nn.Parameter(basis.transpose(0, 1) @ self.bias.detach())
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ self.bias
        return F.linear(x, weight, bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = MeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = MeanZeroOutputLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.token_emb = VocabCenteredEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = VocabCenteredLMHead(self.token_emb)

        self.apply(self._init_weights)

        # Absorb the removed token-embedding row mean into every position so
        # initialization is functionally equivalent to the uncompressed model.
        with torch.no_grad():
            shift = self.lm_head.init_shift
            shift = shift - shift.mean()
            reduced_shift = shift @ self.pos_emb.basis
            self.pos_emb.weight.add_(reduced_shift.unsqueeze(0))
        del self.lm_head.init_shift
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, VocabCenteredEmbedding):
            full_weight = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(
                    module.basis.transpose(0, 1) @ full_weight
                )
        elif isinstance(module, VocabCenteredLMHead):
            full_weight = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(
                    module.basis.transpose(0, 1) @ full_weight
                )
                module.init_shift = full_weight.mean(dim=0)
        elif isinstance(module, MeanZeroEmbedding):
            full_weight = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full_weight @ module.basis)
        elif isinstance(module, MeanZeroOutputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(
                    module.basis.transpose(0, 1) @ full_weight
                )
                module.bias.zero_()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE