MECHANISM: Three-coordinate dual-quadratic lexical manifold

HYPOTHESIS: Replacing each four-scalar token code with a three-scalar code expanded into five features by two normalized quadratic interactions will retain at least 99% accuracy while removing one learned parameter per vocabulary token beyond the qualified 672-parameter backbone.

INTENDED_EDIT: Preserve the qualified attention routing and four-coordinate MLP bias fixing. Challenge the prior assumption that five lexical features require four independent token coordinates by generating those features from three learned coordinates and two input/classifier-specific quadratic lifts.

EVIDENCE: The 672-parameter architecture achieved 99.52%, while changes to attention routing and value transport collapsed accuracy. This motivates preserving those load-bearing mechanisms and testing the previously unchallenged lexical-code dimensionality with a full-width nonlinear lift.

<<<<<<< SEARCH
class QuotientOutputLinear(nn.Module):
    """Linear map into the zero-mean feature subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if out_features < 2:
            raise ValueError("out_features must be at least two")

        self.coeff = nn.Linear(in_features, out_features - 1, bias=bias)

        basis = torch.zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(x) @ self.basis.transpose(0, 1)
=======
class QuotientOutputLinear(nn.Module):
    """Linear map into the zero-mean feature subspace."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        fixed_bias_coeffs: int = 0,
    ):
        super().__init__()
        if out_features < 2:
            raise ValueError("out_features must be at least two")
        if fixed_bias_coeffs < 0 or fixed_bias_coeffs >= out_features - 1:
            raise ValueError("invalid number of fixed bias coefficients")

        self.fixed_bias_coeffs = fixed_bias_coeffs
        self.coeff = nn.Linear(in_features, out_features - 1, bias=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features - 1 - fixed_bias_coeffs))
            if bias
            else None
        )

        basis = torch.zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = (
            None
            if self.bias is None
            else F.pad(self.bias, (0, self.fixed_bias_coeffs))
        )
        return F.linear(x, self.coeff.weight, bias) @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
class FactorizedTokenEmbedding(nn.Module):
    """Three-coordinate codes expanded to a full quotient-space lexical lift."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank < 3 or rank + 2 >= embedding_dim:
            raise ValueError(
                "rank must leave room for two nonlinear zero-mean features"
            )

        self.code = nn.Embedding(num_embeddings, rank)

        basis = torch.zeros(embedding_dim, rank + 2)
        for col in range(rank + 2):
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
            quadratic = torch.cat(
                (
                    code[..., 1:2] * code[..., 2:3],
                    code[..., 2:3] * code[..., :1],
                ),
                dim=-1,
            )
        else:
            quadratic = torch.cat(
                (
                    code[..., :1] * code[..., 1:2],
                    code[..., 1:2] * code[..., 2:3],
                ),
                dim=-1,
            )
        return torch.cat((code, quadratic / rms), dim=-1)

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
        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases and shares
        # one coefficient across its three longest remaining positions.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 8)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases, shares
        # its three longest remaining positions, and separately shares the
        # adjacent pair immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 9)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                second_head_bias,
                second_head_bias[-1:].expand(2),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
=======
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                second_head_bias[:-2],
                second_head_bias[-2:-1].expand(2),
                second_head_bias[-1:].expand(3),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = QuotientOutputLinear(d_ff, d_model)
=======
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=4
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 2
        )
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 3
        )
>>>>>>> REPLACE