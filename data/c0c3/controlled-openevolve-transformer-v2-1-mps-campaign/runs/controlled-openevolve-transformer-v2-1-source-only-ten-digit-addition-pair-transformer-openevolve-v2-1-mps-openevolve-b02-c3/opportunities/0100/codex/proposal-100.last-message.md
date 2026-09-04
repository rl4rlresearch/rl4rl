MECHANISM: Asymmetric homogeneous nonlinear lexical lifts

HYPOTHESIS: A 683-parameter transformer will retain at least 99% accuracy because four learned token coordinates can produce distinct five-dimensional input and classifier geometries through scale-homogeneous quadratic lifts, while preserving all three load-bearing final-normalization scales and the qualified 797-parameter attention layout.

INTENDED_EDIT: Reconstruct the qualified 797-parameter relative-bias and classifier configuration, then replace the shared five-coordinate token table with four-coordinate codes whose input and output paths use different RMS-normalized quadratic fifth coordinates.

EVIDENCE: The fixed shared quadratic lift reached only 41.97%, despite the linear rank-four model reaching 95.12%, while removing one final LayerNorm scale separately collapsed accuracy to 10.87%. This patch tests whether the failure came from forcing both lexical paths onto the same poorly scaled nonlinear chart: it keeps five-dimensional geometry on both paths, preserves all three normalization scales, and starts from the 797-parameter attention design that achieved 99.83%.

<<<<<<< SEARCH
class FactorizedTokenEmbedding(nn.Module):
    """Fixed-subspace low-rank token map shared with the output classifier."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank < 1 or rank >= embedding_dim:
            raise ValueError("rank must be between one and embedding_dim - 1")

        self.code = nn.Embedding(num_embeddings, rank)

        basis = torch.zeros(embedding_dim, rank)
        for col in range(rank):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def projection_weight(self) -> torch.Tensor:
        return self.basis

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.linear(self.code(tokens), self.projection_weight())

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.projection_weight()
        latent = F.linear(x, weight.transpose(0, 1))
        return F.linear(latent, self.code.weight)
=======
class FactorizedTokenEmbedding(nn.Module):
    """Four-coordinate codes with distinct full-rank input and output lifts."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank < 2 or rank + 1 >= embedding_dim:
            raise ValueError(
                "rank must leave room for one nonlinear zero-mean feature"
            )

        self.code = nn.Embedding(num_embeddings, rank)

        basis = torch.zeros(embedding_dim, rank + 1)
        for col in range(rank + 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def projection_weight(self) -> torch.Tensor:
        return self.basis

    @staticmethod
    def _lift(code: torch.Tensor, classifier: bool) -> torch.Tensor:
        rms = code.square().mean(dim=-1, keepdim=True).add(1e-8).sqrt()
        if classifier:
            quadratic = code[..., -2:-1] * code[..., -1:] / rms
        else:
            quadratic = code[..., :1] * code[..., 1:2] / rms
        return torch.cat((code, quadratic), dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        code = self._lift(self.code(tokens), classifier=False)
        return F.linear(code, self.projection_weight())

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.projection_weight()
        latent = F.linear(x, weight.transpose(0, 1))
        classifier_code = self._lift(self.code.weight, classifier=True)
        return F.linear(latent, classifier_code)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Distance zero cancels inside softmax, and the uniquely sparse
        # maximum distance is fixed at zero for both heads.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 2))
=======
        # Distance zero and the two longest distances are fixed for both
        # heads; the second head's next two longest biases are also fixed.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * (max_seq_len - 3) - 2)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        learned_relative_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 1),
            ),
            dim=1,
        )
=======
        relative_bias_coeff = F.pad(self.relative_bias, (0, 2)).view(
            self.n_head, -1
        )
        learned_relative_bias = torch.cat(
            (
                relative_bias_coeff,
                relative_bias_coeff.new_zeros(self.n_head, 2),
            ),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 2
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = ClassifierAwareLayerNorm(
            cfg.d_model, self.token_emb.code.embedding_dim + 1
        )
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 4)
        )
=======
        self.ln_f = ClassifierAwareLayerNorm(cfg.d_model, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        final_bias = F.pad(self.final_bias, (0, 4))
        x = x + final_bias @ self.token_emb.basis.transpose(0, 1)
        logits = self.token_emb.logits(x)
=======
        x = self.ln_f(x)
        logits = self.token_emb.logits(x)
>>>>>>> REPLACE