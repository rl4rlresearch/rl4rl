MECHANISM: Mid-pair tied token-code subspace

HYPOTHESIS: Because residual coordinates two and three already share an MLP input vector and final-LayerNorm scale successfully, sharing their learned token embedding/readout coefficient will reduce the model by one parameter per vocabulary token while retaining at least 99% accuracy.

INTENDED_EDIT: Store six rather than seven gauge-fixed token contrasts, reconstruct embedding coordinates two and three from one learned column, and initialize that column from their mean; the constraint affects both input representation and the tied language-model classifier.

EVIDENCE: Mid-pair final-LayerNorm scale sharing achieved 99.75%, and disjoint MLP input pairing achieved 99.93%. This challenges the old assumption that every token needs distinct coefficients on all residual channels, while preserving the independent attention value spaces whose cross-head sharing failed at 15.71%.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with each token's scalar row offset fixed at zero."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim - 1)
        )

        # Preserve the RNG stream of nn.Embedding's constructor.
        torch.empty(num_embeddings, embedding_dim).normal_()

    def initialize_from_full_normal(self) -> None:
        full_weight = self.weight.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.weight.copy_(
                full_weight[:, :-1] - full_weight[:, -1:]
            )

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 1))
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with a shared mid-pair and fixed row-offset gauge."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim - 2)
        )

        # Preserve the RNG stream of nn.Embedding's constructor.
        torch.empty(num_embeddings, embedding_dim).normal_()

    def initialize_from_full_normal(self) -> None:
        full_weight = self.weight.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        contrasts = full_weight[:, :-1] - full_weight[:, -1:]
        with torch.no_grad():
            self.weight.copy_(
                torch.cat(
                    [
                        contrasts[:, :2],
                        contrasts[:, 2:4].mean(dim=-1, keepdim=True),
                        contrasts[:, 4:],
                    ],
                    dim=-1,
                )
            )

    def full_weight(self) -> torch.Tensor:
        return torch.cat(
            [
                self.weight[:, :3],
                self.weight[:, 2:3],
                self.weight[:, 3:],
                self.weight.new_zeros(self.num_embeddings, 1),
            ],
            dim=-1,
        )
>>>>>>> REPLACE