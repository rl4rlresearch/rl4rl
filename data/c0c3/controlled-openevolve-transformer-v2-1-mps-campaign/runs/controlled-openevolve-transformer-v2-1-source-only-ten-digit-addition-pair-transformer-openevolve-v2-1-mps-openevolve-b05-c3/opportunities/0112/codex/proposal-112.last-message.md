MECHANISM: Four-way final-LayerNorm bias tying with per-token embedding quotient

HYPOTHESIS: Extending the qualified three-way final-bias tie to four channels will reduce the model from 1,129 to 1,128 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, and token-representation capacity remain unchanged.

INTENDED_EDIT: Adopt the verified per-token embedding quotient and scalar final scale, then reconstruct the zero-sum final bias from four learned coordinates with its final four coordinates tied.

EVIDENCE: Three-way final-bias tying achieved 99.83% accuracy with 1,129 parameters after two-way tying achieved 99.90% with 1,130; this supports one further isolated bias tie while preserving components whose factorization or sharing caused severe accuracy collapse.

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
    """Final LayerNorm with its last three channel scales tied."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 2))

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight[-1:].expand(2)))

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
    """Final LayerNorm with shared scale and four-way-tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(d_model - 4))

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def full_bias(self) -> torch.Tensor:
        anchored = torch.cat((self.bias, self.bias.new_zeros(4)))
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