MECHANISM: Row-centered tied-embedding contrast classifier

HYPOTHESIS: Centering the final hidden state before the tied output projection makes each token embedding’s scalar row offset exactly unobservable, allowing 113 embedding parameters to be removed while the remaining seven-dimensional learned contrasts retain at least 99% accuracy; the resulting model has 1,284 parameters.

INTENDED_EDIT: Store seven within-row embedding differences per token instead of one globally gauge-fixed flattened table, center the learned final representation before computing logits, and preserve full-width AdamW geometry with one virtual reference coordinate per token.

EVIDENCE: The 1,397-parameter model reached 99.93%, while optimizer-preserving quotients repeatedly retained high accuracy. Unlike removing load-bearing LayerNorm scales, this keeps the final affine LayerNorm and instead challenges the assumption that token-specific embedding common modes are needed by making the classifier operate entirely on learned residual contrasts.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with one global all-entries offset fixed at zero."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings * embedding_dim - 1)
        )
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with each token's scalar row offset fixed at zero."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        flat = full_weight.reshape(-1)
        with torch.no_grad():
            self.weight.copy_(flat[:-1] - flat[-1])

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 1)).view(
            self.num_embeddings, self.embedding_dim
        )
=======
        with torch.no_grad():
            self.weight.copy_(
                full_weight[:, :-1] - full_weight[:, -1:]
            )

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding.full_weight())


class GaugeFixedResidualProjection(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only learned within-coordinate contrasts are observable by the
        # row-gauge-fixed tied classifier.
        x = x - x.mean(dim=-1, keepdim=True)
        return F.linear(x, self.embedding.full_weight())


class GaugeFixedResidualProjection(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauge_specs = [
            (model.token_emb.weight, 0)
        ] + [
=======
        self.gauge_specs = [
            (model.token_emb.weight, 1)
        ] + [
>>>>>>> REPLACE