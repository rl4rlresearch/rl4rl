MECHANISM: Seventh-longest tolerant-head relative-bias fixing

HYPOTHESIS: A 680-parameter transformer will retain at least 99% accuracy because fixing the second head’s seventh-longest relative-position bias preserves the qualified asymmetric nonlinear lexical lifts, all three load-bearing final-normalization scales, and every first-head routing coefficient.

INTENDED_EDIT: Reconstruct the qualified 681-parameter nonlinear lexical design, then remove one additional relative-bias coefficient from the more heavily pruned second attention head.

EVIDENCE: The 681-parameter design achieved 99.96% after sequentially pruning the second head through its sixth-longest bias, while pruning the corresponding first-head coefficient previously fell to 92.07%; this supports one more adjacent second-head reduction.

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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.linear(self.code(tokens), self.basis)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        latent = F.linear(x, self.basis.transpose(0, 1))
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
        # Distance zero and the two sparsest maximum distances are fixed
        # for both heads; the last head's next-sparsest bias is also fixed.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * (max_seq_len - 3) - 1)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads; the second head's next five longest biases are also fixed.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * (max_seq_len - 3) - 5)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_coeff = F.pad(self.relative_bias, (0, 1)).view(
            self.n_head, -1
        )
=======
        relative_bias_coeff = F.pad(self.relative_bias, (0, 5)).view(
            self.n_head, -1
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