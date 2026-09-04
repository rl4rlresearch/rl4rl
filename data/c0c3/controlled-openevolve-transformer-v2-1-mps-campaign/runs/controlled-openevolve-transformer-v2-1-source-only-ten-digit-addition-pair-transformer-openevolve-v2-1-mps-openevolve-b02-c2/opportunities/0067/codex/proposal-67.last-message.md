MECHANISM: Fourth token–position embedding translation gauge

HYPOTHESIS: Extending the verified 1,615-parameter design to fix positional-origin coordinate 3 will yield 1,614 parameters and at least 99% accuracy, because the transformation preserves initialized hidden inputs and adds only a uniform shift to tied-head logits.

INTENDED_EDIT: Reproduce the verified three-key-row and first-MLP-row quotients, then absorb the first four positional-origin coordinates into the corresponding token-embedding coordinates and omit those four positional parameters.

EVIDENCE: The three-gauge 1,615-parameter design achieved 99.84% accuracy; the second and third positional gauges also independently preserved at least 99%, directly supporting one more application of the same symmetry.

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
    """Positional embedding with four token-position translation gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[4:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(4), self.weight))
>>>>>>> REPLACE

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with two LayerNorm-induced key-weight gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with three LayerNorm-induced key-weight gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        retained_weight = torch.cat(
            (
                linear.weight[:key_start],
                linear.weight[key_start + 1 : second_head_start],
                linear.weight[second_head_start + 1 :],
            ),
            dim=0,
        )
=======
        retained_weight = torch.cat(
            (
                linear.weight[:key_start],
                linear.weight[key_start + 2 : second_head_start],
                linear.weight[second_head_start + 1 :],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_key_weight = (
            linear.weight[[key_start, second_head_start]] * ln_weight
        )
=======
        scaled_key_weight = (
            linear.weight[[key_start, key_start + 1, second_head_start]]
            * ln_weight
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        second_retained_start = self.second_key_row - 1
        full_weight = torch.cat(
            (
                self.weight[: self.key_start],
                key_weight[:1],
                self.weight[self.key_start : second_retained_start],
                key_weight[1:],
                self.weight[second_retained_start:],
            ),
            dim=0,
        )
=======
        second_retained_start = self.second_key_row - 2
        full_weight = torch.cat(
            (
                self.weight[: self.key_start],
                key_weight[:2],
                self.weight[self.key_start : second_retained_start],
                key_weight[2:],
                self.weight[second_retained_start:],
            ),
            dim=0,
        )
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

        # Retain the qualified bias layout, quotient the effective attention
        # offset, and fix one key-weight gauge in each attention head.
=======
        # Fix four exact token/position translation gauges while preserving
        # the initialized hidden inputs and output probabilities.
        with torch.no_grad():
            for coordinate in range(4):
                offset = self.pos_emb.weight[0, coordinate].detach().clone()
                self.token_emb.weight[:, coordinate].add_(offset)
                self.pos_emb.weight[:, coordinate].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)

        # Retain the qualified bias and projection layout, fix three key-row
        # gauges, and quotient one independently biased MLP input row.
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.proj = CompactSharedProjection(
                block.attn.proj,
                compact_qkv.value_bias,
            )

    @staticmethod
=======
            block.attn.proj = CompactSharedProjection(
                block.attn.proj,
                compact_qkv.value_bias,
            )
            block.mlp.fc1 = CompactFirstLinearRow(
                block.mlp.fc1,
                block.ln2.weight,
            )

    @staticmethod
>>>>>>> REPLACE