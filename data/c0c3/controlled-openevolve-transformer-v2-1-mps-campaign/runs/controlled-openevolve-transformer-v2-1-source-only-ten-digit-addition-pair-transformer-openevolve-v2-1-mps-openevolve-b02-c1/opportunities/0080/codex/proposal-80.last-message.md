MECHANISM: Six-dimensional tied lexical bottleneck

HYPOTHESIS: Constraining each tied token embedding to six learned contrast coordinates will remove one parameter per vocabulary token while retaining at least 99% accuracy, because rank-two content routing already achieved 99.93% and the independently learned relative-position pathways—not maximal content width—were the demonstrated load-bearing mechanism.

INTENDED_EDIT: Replace the seven-degree token embedding with a six-degree learned representation whose final two nonreference channels share one learned coordinate; preserve the eight-dimensional residual stream, tied learned classifier, causal attention, and quotient optimization.

EVIDENCE: Reducing query/key routing from rank three to rank two retained 99.93%, while rank one failed and relative-bias replacement failed completely. This patch therefore preserves the proven rank-two attention and positional tables while challenging the separate assumption that every vocabulary token requires all seven gauge-fixed lexical degrees.

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
    """Tied embedding with a six-dimensional learned lexical bottleneck."""

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
                        contrasts[:, :-2],
                        0.5 * (
                            contrasts[:, -2:-1]
                            + contrasts[:, -1:]
                        ),
                    ],
                    dim=-1,
                )
            )

    def full_weight(self) -> torch.Tensor:
        return torch.cat(
            [
                self.weight[:, :-1],
                self.weight[:, -1:].expand(-1, 2),
                self.weight.new_zeros(self.num_embeddings, 1),
            ],
            dim=-1,
        )
>>>>>>> REPLACE