MECHANISM: Orthonormal MLP residual-bias gauge quotient

HYPOTHESIS: Applying a seven-dimensional zero-mean parameterization to `fc2.bias` on the verified five-position-gauge baseline will produce a 1,612-parameter model with at least 99% accuracy, because the removed uniform residual shift is canceled by the final LayerNorm.

INTENDED_EDIT: Reproduce the verified fifth token–position gauge, then replace the eight-parameter `fc2` bias with seven Helmert coordinates that reconstruct a zero-mean bias.

EVIDENCE: The five-position-gauge design achieved 99.94% accuracy at 1,613 parameters, while a sixth positional gauge fell to 94.83%; the qualified designs already exploit the analogous zero-mean residual-bias invariance in the attention projection, motivating this independent exact quotient.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Positional embedding with four token-position translation gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[4:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(4), self.weight))
=======
class CompactPositionEmbedding(nn.Module):
    """Positional embedding with five token-position translation gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[5:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(5), self.weight))
>>>>>>> REPLACE

<<<<<<< SEARCH
        return F.linear(x, full_weight, self.bias)


class MLP(nn.Module):
=======
        return F.linear(x, full_weight, self.bias)


class CompactResidualLinear(nn.Module):
    """Linear layer with its residual-uniform bias direction fixed."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight

        width = linear.out_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)

        centered_bias = linear.bias - linear.bias.mean()
        self.bias = nn.Parameter(
            (basis.transpose(0, 1) @ centered_bias).detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = self.bias_basis @ self.bias
        return F.linear(x, self.weight, full_bias)


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix four exact token/position translation gauges while preserving
        # the initialized hidden inputs and output probabilities.
        with torch.no_grad():
            for coordinate in range(4):
                offset = self.pos_emb.weight[0, coordinate].detach().clone()
                self.token_emb.weight[:, coordinate].add_(offset)
                self.pos_emb.weight[:, coordinate].sub_(offset)
=======
        # Fix five exact token/position translation gauges while preserving
        # the initialized hidden inputs and output probabilities.
        with torch.no_grad():
            for coordinate in range(5):
                offset = self.pos_emb.weight[0, coordinate].detach().clone()
                self.token_emb.weight[:, coordinate].add_(offset)
                self.pos_emb.weight[:, coordinate].sub_(offset)
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.mlp.fc1 = CompactFirstLinearRow(
                block.mlp.fc1,
                block.ln2.weight,
            )
=======
            block.mlp.fc1 = CompactFirstLinearRow(
                block.mlp.fc1,
                block.ln2.weight,
            )
            block.mlp.fc2 = CompactResidualLinear(block.mlp.fc2)
>>>>>>> REPLACE