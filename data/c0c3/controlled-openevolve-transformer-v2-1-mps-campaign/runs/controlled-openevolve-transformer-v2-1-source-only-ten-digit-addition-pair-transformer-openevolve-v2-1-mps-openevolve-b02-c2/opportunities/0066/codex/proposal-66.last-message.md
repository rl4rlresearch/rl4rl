MECHANISM: Third token–position embedding translation gauge

HYPOTHESIS: Extending the verified 1,616-parameter design to fix positional-embedding origin coordinate 2 will yield 1,615 parameters and at least 99% accuracy, because the same coordinate-wise translation leaves initialized hidden inputs unchanged and shifts every output logit equally.

INTENDED_EDIT: Reproduce the qualified first MLP-row quotient and two positional gauges, then absorb `pos_emb[0, 2]` into token-embedding coordinate 2 and omit the first three flattened positional parameters.

EVIDENCE: The first LayerNorm–MLP row quotient achieved 99.84% at 1,617 parameters, and adding the second exact token–position translation gauge produced 99.93% at 1,616; extending that qualified embedding symmetry by one coordinate is the smallest directly supported reduction.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Positional embedding with one token-position translation gauge fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[1:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(1), self.weight))
=======
class CompactPositionEmbedding(nn.Module):
    """Positional embedding with three token-position translation gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[3:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(3), self.weight))
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
=======
class CompactFirstLinearRow(nn.Module):
    """Linear layer with one LayerNorm input-direction gauge fixed."""

    def __init__(self, linear: nn.Linear, ln_weight: nn.Parameter):
        super().__init__()
        self.ln_weight = ln_weight
        self.weight = nn.Parameter(linear.weight[1:].detach().clone())
        self.bias = linear.bias

        width = linear.in_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("row_basis", basis, persistent=False)

        scaled_row = linear.weight[0] * ln_weight
        centered_row = scaled_row - scaled_row.mean()
        self.row_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_row).detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_row = self.row_basis @ self.row_weight
        first_row = scaled_row / self.ln_weight
        full_weight = torch.cat((first_row.unsqueeze(0), self.weight), dim=0)
        return F.linear(x, full_weight, self.bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix one exact token/position translation gauge while preserving the
        # initialized hidden inputs and output probabilities.
        with torch.no_grad():
            offset = self.pos_emb.weight[0, 0].detach().clone()
            self.token_emb.weight[:, 0].add_(offset)
            self.pos_emb.weight[:, 0].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)
=======
        # Fix three exact token/position translation gauges while preserving
        # the initialized hidden inputs and output probabilities.
        with torch.no_grad():
            for coordinate in range(3):
                offset = self.pos_emb.weight[0, coordinate].detach().clone()
                self.token_emb.weight[:, coordinate].add_(offset)
                self.pos_emb.weight[:, coordinate].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified bias layout, quotient the effective attention
        # offset, and fix three independent key-weight gauges.
        for block in self.blocks:
            compact_qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
                block.ln1.weight,
            )
            block.attn.qkv = compact_qkv
            block.attn.proj = CompactSharedProjection(
                block.attn.proj,
                compact_qkv.value_bias,
            )
=======
        # Retain the qualified bias and projection layout, fix three key-row
        # gauges, and quotient one independently biased MLP input row.
        for block in self.blocks:
            compact_qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
                block.ln1.weight,
            )
            block.attn.qkv = compact_qkv
            block.attn.proj = CompactSharedProjection(
                block.attn.proj,
                compact_qkv.value_bias,
            )
            block.mlp.fc1 = CompactFirstLinearRow(
                block.mlp.fc1,
                block.ln2.weight,
            )
>>>>>>> REPLACE