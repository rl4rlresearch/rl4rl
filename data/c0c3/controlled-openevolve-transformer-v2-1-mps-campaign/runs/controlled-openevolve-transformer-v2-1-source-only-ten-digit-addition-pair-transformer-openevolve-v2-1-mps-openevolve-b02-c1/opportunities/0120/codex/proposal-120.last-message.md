MECHANISM: Learned four-dimensional token-code bottleneck

HYPOTHESIS: Replacing each seven-coordinate token embedding with a learned four-dimensional code and a learned shared basis will reduce parameters by `3 * vocab_size - 12` while retaining at least 99% accuracy, because it preserves both specialized attention heads and the verified relative-position routing while allowing every symbol representation and output logit to remain learned.

INTENDED_EDIT: Factor the tied input/output embedding through a gauge-fixed rank-four basis, decode logits through that factorization without materializing the full table, and optimize both identifiable factors directly with AdamW.

EVIDENCE: Replacing positional routing with Gaussian bands collapsed accuracy to 2.09%, and sharing head value maps collapsed it to 15.71%, showing that routing and head specialization are load-bearing. In contrast, paired MLP input coordinates retained 99.93%, supporting a different hypothesis: the model needs its existing internal attention computation, but not seven independent learned coordinates for every token.

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

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied rank-four token codes in an identifiable learned subspace."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.code_dim = 4
        if embedding_dim - 1 < self.code_dim:
            raise ValueError("embedding_dim must provide four gauge-fixed coordinates")

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, self.code_dim)
        )
        self.basis_tail = nn.Parameter(
            torch.empty(
                self.code_dim,
                embedding_dim - 1 - self.code_dim,
            )
        )

        # Preserve the RNG stream of nn.Embedding's constructor.
        torch.empty(num_embeddings, embedding_dim).normal_()

    def initialize_from_full_normal(self) -> None:
        full_weight = self.weight.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        reduced_weight = full_weight[:, :-1] - full_weight[:, -1:]
        basis_scale = 0.02 * math.sqrt(self.code_dim)
        with torch.no_grad():
            self.weight.copy_(reduced_weight[:, : self.code_dim])
            self.basis_tail.copy_(
                full_weight[
                    : self.code_dim,
                    self.code_dim : -1,
                ]
                / basis_scale
            )

    def reduced_basis(self) -> torch.Tensor:
        prefix = torch.eye(
            self.code_dim,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        return torch.cat([prefix, self.basis_tail], dim=-1)

    def full_weight(self) -> torch.Tensor:
        reduced_weight = torch.matmul(
            self.weight, self.reduced_basis()
        )
        return F.pad(reduced_weight, (0, 1))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        codes = F.embedding(idx, self.weight)
        reduced = F.linear(
            codes, self.reduced_basis().transpose(0, 1)
        )
        return F.pad(reduced, (0, 1))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        code_state = F.linear(
            x[..., :-1], self.reduced_basis()
        )
        return F.linear(code_state, self.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only learned within-coordinate contrasts are observable by the
        # row-gauge-fixed tied classifier.
        x = x - x.mean(dim=-1, keepdim=True)
        return F.linear(x, self.embedding.full_weight())
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only learned within-coordinate contrasts are observable by the
        # row-gauge-fixed tied classifier.
        return self.embedding.decode(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauge_specs = [
            (model.token_emb.weight, 1),
            (model.ln_f.bias, 0),
        ] + [
=======
        self.gauge_specs = [
            (model.ln_f.bias, 0),
        ] + [
>>>>>>> REPLACE