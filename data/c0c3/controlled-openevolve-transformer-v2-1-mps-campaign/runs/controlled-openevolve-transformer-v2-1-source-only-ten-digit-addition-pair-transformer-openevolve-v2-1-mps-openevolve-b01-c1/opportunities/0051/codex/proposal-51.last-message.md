MECHANISM: One-coordinate absolute-position bottleneck

HYPOTHESIS: Restricting each learned positional vector from seven to six mean-free coordinates will reduce the model by `INPUT_LEN` parameters while retaining at least 99% accuracy, because the current model reaches 99.92% and this preserves learned absolute positioning, all initialization draws, and six independent positional directions.

INTENDED_EDIT: Remove one orthogonal coordinate from `MeanFreePositionEmbedding`, projecting the unchanged full-width initialization draw into the retained six-dimensional subspace.

EVIDENCE: The 1,578-parameter design achieves 99.92%, while the attempted wholesale replacement of absolute positions with relative logits could not be verified; this smaller, isolated test targets the same large parameter source without changing attention behavior.

<<<<<<< SEARCH
class MeanFreePositionEmbedding(nn.Module):
    """Learned positional vectors modulo LayerNorm-invariant constant offsets."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 1))

        basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for j in range(embedding_dim - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
=======
class MeanFreePositionEmbedding(nn.Module):
    """Learned positional vectors in a six-dimensional mean-free subspace."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 2))

        basis = torch.zeros(embedding_dim, embedding_dim - 2)
        for j in range(embedding_dim - 2):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
>>>>>>> REPLACE