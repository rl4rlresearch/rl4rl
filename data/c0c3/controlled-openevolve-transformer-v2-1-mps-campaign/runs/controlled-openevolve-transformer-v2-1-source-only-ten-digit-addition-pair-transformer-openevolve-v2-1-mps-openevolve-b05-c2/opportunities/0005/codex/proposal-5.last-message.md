MECHANISM: Residual gauge compression with one-coordinate value-bias anchoring

HYPOTHESIS: Compressing the qualified model into LayerNorm-invisible mean-zero subspaces and retaining seven of eight learned value-bias coordinates will achieve at least 99% accuracy with 1590 parameters.

INTENDED_EDIT: Apply the verified mean-zero positional/residual parameterization and key-bias removal, then fix one functionally absorbable value-bias coordinate to zero while retaining the other seven.

EVIDENCE: The full mean-zero design achieved 99.96% accuracy with 1591 parameters, while removing all eight value-bias coordinates collapsed accuracy; retaining seven tests the smallest conservative reduction beyond the qualified design.

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
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        value_bias = F.pad(self.v_bias, (0, 1))
        qkv_bias = torch.cat(
            (self.q_bias, self.qkv.weight.new_zeros(d_model), value_bias)
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = MeanZeroOutputLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
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
        if isinstance(module, MeanZeroEmbedding):
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