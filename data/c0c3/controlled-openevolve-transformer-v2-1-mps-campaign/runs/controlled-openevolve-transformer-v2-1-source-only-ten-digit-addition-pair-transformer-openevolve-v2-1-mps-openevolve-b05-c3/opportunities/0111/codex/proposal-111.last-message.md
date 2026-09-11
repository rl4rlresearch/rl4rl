MECHANISM: Three-way final-bias tying with per-token embedding quotient

HYPOTHESIS: Extending the verified two-way final-bias tie to three channels will reduce the qualified 1,130-parameter model to 1,129 parameters while retaining at least 99% accuracy, because the learned shared normalization scale and all attention/value capacity remain unchanged.

INTENDED_EDIT: Adopt the verified per-token embedding quotient and scalar final scale, then reconstruct the zero-sum final bias from five learned coordinates with its final three coordinates tied.

EVIDENCE: The two-way final-bias tie achieved 99.90% with 1,130 parameters, while relocating the adaptive scale collapsed to 88.17% and sharing value encoders collapsed to 6.31%; this motivates one further isolated final-bias tie while preserving those sensitive components.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Embedding):
    """Tied embedding represented modulo one global scalar shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the constructor RNG used by the original full embedding.
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings * embedding_dim - 1)
        )

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat((self.weight, self.weight.new_zeros(1)))
        return flat.view(self.num_embeddings, self.embedding_dim)
=======
class GaugeFixedEmbedding(nn.Embedding):
    """Tied embedding modulo one hidden-coordinate shift per token."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the constructor RNG used by the original full embedding.
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim - 1)
        )

    def full_weight(self) -> torch.Tensor:
        return torch.cat(
            (
                self.weight,
                self.weight.new_zeros(self.num_embeddings, 1),
            ),
            dim=-1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with its last seven channel scales tied."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 6))

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight[-1:].expand(6)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            self.bias,
            self.eps,
        )
=======
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and three-way-tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(d_model - 3))

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def full_bias(self) -> torch.Tensor:
        anchored = torch.cat((self.bias, self.bias.new_zeros(3)))
        return anchored - anchored.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            self.full_bias(),
            self.eps,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedEmbedding):
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                full.sub_(full[-1, -1].clone())
                module.weight.copy_(full.reshape(-1)[:-1])
=======
        if isinstance(module, GaugeFixedEmbedding):
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[:, -1:].clone()
                full[:, :-1].sub_(omitted)
                full[:, -1].zero_()
                module.weight.copy_(full[:, :-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeTiedHead):
            embedding = module._embedding
            full = embedding.weight.new_empty(
                embedding.num_embeddings, embedding.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                full.sub_(full[-1, -1].clone())
                embedding.weight.copy_(full.reshape(-1)[:-1])
=======
        elif isinstance(module, GaugeTiedHead):
            embedding = module._embedding
            full = embedding.weight.new_empty(
                embedding.num_embeddings, embedding.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[:, -1:].clone()
                full[:, :-1].sub_(omitted)
                full[:, -1].zero_()
                embedding.weight.copy_(full[:, :-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
    relative_bias_gauge_parameters = [
        block.attn.relative_bias for block in model.blocks
    ]
    last_coordinate_gauge_parameters = [
        *embedding_gauge_parameters,
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
    ]
=======
    relative_bias_gauge_parameters = [
        block.attn.relative_bias for block in model.blocks
    ]
    final_bias_gauge_parameters = [
        model.ln_f.bias
    ]
    last_coordinate_gauge_parameters = [
        *embedding_gauge_parameters,
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
        *final_bias_gauge_parameters,
    ]
>>>>>>> REPLACE