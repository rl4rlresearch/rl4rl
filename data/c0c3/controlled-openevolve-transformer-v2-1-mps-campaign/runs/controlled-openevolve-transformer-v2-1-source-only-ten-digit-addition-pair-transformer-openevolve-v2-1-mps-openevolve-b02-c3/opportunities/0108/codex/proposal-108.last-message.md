MECHANISM: Three-way tolerant-head relative-bias sharing

HYPOTHESIS: A 677-parameter transformer will retain at least 99% accuracy because extending the qualified second head’s two-way adjacent bias sharing to three adjacent long-distance biases preserves a learned routing signal while leaving the first head and all normalization scales unchanged.

INTENDED_EDIT: Reconstruct the qualified nonlinear lexical lifts and relative-bias layout, then remove one parameter by sharing the second head’s learned coefficient across its three longest remaining bias positions.

EVIDENCE: The 678-parameter two-way sharing design achieved 99.97%, whereas fixing the additional bias at zero and anchoring a QKV coefficient collapsed accuracy; extending learned sharing is the narrowest adjacent reduction supported by that evidence.

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
        # Distance zero and the uniquely sparse maximum distance are fixed.
        # The last head's next-longest distance is also fixed at zero.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * (max_seq_len - 2) - 1)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases and shares
        # one coefficient across its three longest remaining positions.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 8)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_coeff = F.pad(self.relative_bias, (0, 1)).view(
            self.n_head, -1
        )
        learned_relative_bias = torch.cat(
            (
                relative_bias_coeff,
                relative_bias_coeff.new_zeros(self.n_head, 1),
            ),
            dim=1,
        )
=======
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                second_head_bias,
                second_head_bias[-1:].expand(2),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
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
=======
        self.ln_f = ClassifierAwareLayerNorm(cfg.d_model, cfg.d_model)
>>>>>>> REPLACE