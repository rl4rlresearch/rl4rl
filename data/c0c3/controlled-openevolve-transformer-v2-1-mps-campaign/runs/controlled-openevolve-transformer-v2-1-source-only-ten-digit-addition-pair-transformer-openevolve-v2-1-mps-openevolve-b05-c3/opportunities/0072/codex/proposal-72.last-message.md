MECHANISM: Four-coordinate pre-attention LayerNorm-scale quotient

HYPOTHESIS: Extending the qualified three-coordinate `ln1` scale quotient to four coordinates will produce 1,316 learned parameters and retain at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while four adaptive scales remain.

INTENDED_EDIT: Adopt the qualified global tied-embedding and complete attention/MLP projection gauges, remove `ln1` bias, and fix its final four scale coordinates at one while preserving full-space initialization and gauge-aware optimization.

EVIDENCE: The three-coordinate quotient achieved 99.47% accuracy with 1,317 parameters, after the one- and two-coordinate quotients also exceeded 99%; this incremental extension directly probes the optimization-sensitive boundary exposed by the failed all-eight-scale removal.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Embedding):
    """Embedding vectors represented modulo a shared channel shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume exactly the constructor RNG used by the original embedding.
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 1))

    def full_weight(self) -> torch.Tensor:
        zero = self.weight.new_zeros(self.num_embeddings, 1)
        return torch.cat((self.weight, zero), dim=-1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            idx,
            self.full_weight(),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class FixedRouteValueLinear(nn.Linear):
=======
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

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            idx,
            self.full_weight(),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with four scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 4))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )


class FixedRouteValueLinear(nn.Linear):
>>>>>>> REPLACE

<<<<<<< SEARCH
class FixedRouteValueLinear(nn.Linear):
    """Value projection for content-independent learned attention."""

    def __init__(self, d_model: int):
        # Preserve the original QKV constructor's RNG consumption.
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.out_features = d_model
        self.weight = nn.Parameter(torch.empty(d_model, d_model))
        self.bias = None


class GaugeFixedProjectionLinear(nn.Linear):
=======
class FixedRouteValueLinear(nn.Linear):
    """Value projection for content-independent learned attention."""

    def __init__(self, d_model: int):
        # Preserve the original QKV constructor's RNG consumption.
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.out_features = d_model
        self.weight = nn.Parameter(torch.empty(d_model, d_model))
        self.bias = None


class GaugeTiedHead(nn.Linear):
    """Parameter-free view of the globally gauge-fixed tied embedding."""

    def __init__(self, embedding: GaugeFixedEmbedding):
        # Preserve the original tied Linear constructor's RNG consumption.
        super().__init__(
            embedding.embedding_dim,
            embedding.num_embeddings,
            bias=False,
        )
        self.weight = None
        object.__setattr__(self, "_embedding", embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._embedding.full_weight())


class GaugeFixedProjectionLinear(nn.Linear):
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedProjectionLinear(nn.Linear):
    """Attention projection with two common-output-shift coordinates fixed."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.missing_start = (d_model - 1) * d_model
        self.weight = nn.Parameter(torch.empty(d_model * d_model - 2))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(2),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_model)


class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with eleven common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 11
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 11))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(11),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)
=======
class GaugeFixedProjectionLinear(nn.Linear):
    """Attention projection with all common-output shifts fixed."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.missing_start = (d_model - 1) * d_model
        self.weight = nn.Parameter(torch.empty((d_model - 1) * d_model))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight,
                self.weight.new_zeros(self.d_model),
            )
        )
        return flat.view(self.d_model, self.d_model)


class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with all common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = (d_model - 1) * d_ff
        self.weight = nn.Parameter(torch.empty((d_model - 1) * d_ff))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight,
                self.weight.new_zeros(self.d_ff),
            )
        )
        return flat.view(self.d_model, self.d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
=======
        self.ln1 = GaugeFixedScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        self.token_emb = GaugeFixedEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with the reconstructed full input embedding.
        self.lm_head = GaugeTiedHead(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedEmbedding):
            # Draw the original full tensor so every later initialization keeps
            # the qualified RNG stream, then choose the equivalent gauge whose
            # final coordinate is zero.
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full[:, :-1] - full[:, -1:])
=======
        if isinstance(module, GaugeFixedEmbedding):
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                full.sub_(full[-1, -1].clone())
                module.weight.copy_(full.reshape(-1)[:-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, FixedRouteValueLinear):
            d_model = module.d_model
            full = module.weight.new_empty(3 * d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full[2 * d_model :])
        elif isinstance(module, GaugeFixedProjectionLinear):
=======
        elif isinstance(module, FixedRouteValueLinear):
            d_model = module.d_model
            full = module.weight.new_empty(3 * d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full[2 * d_model :])
        elif isinstance(module, GaugeTiedHead):
            embedding = module._embedding
            full = embedding.weight.new_empty(
                embedding.num_embeddings, embedding.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                full.sub_(full[-1, -1].clone())
                embedding.weight.copy_(full.reshape(-1)[:-1])
        elif isinstance(module, GaugeFixedProjectionLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedProjectionLinear):
            d_model = module.d_model
            full = module.weight.new_empty(d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, :2].clone()
                full[:, :2].sub_(omitted)
                full[-1, :2].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 2 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
=======
        elif isinstance(module, GaugeFixedProjectionLinear):
            d_model = module.d_model
            full = module.weight.new_empty(d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1].clone()
                full.sub_(omitted)
                full[-1].zero_()
                module.weight.copy_(full[:-1].reshape(-1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
            full = module.weight.new_empty(
                module.d_model, module.d_ff
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, -11:].clone()
                full[:, -11:].sub_(omitted)
                full[-1, -11:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 11 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
=======
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
            full = module.weight.new_empty(
                module.d_model, module.d_ff
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1].clone()
                full.sub_(omitted)
                full[-1].zero_()
                module.weight.copy_(full[:-1].reshape(-1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    missing_start = (d_model - 1) * d_model
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(2),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_model)
    full_grad[-1, :2] = -full_grad[:-1, :2].sum(dim=0)
    return full_grad
=======
def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    full_grad = torch.cat(
        (
            parameter.grad,
            parameter.grad.new_zeros(d_model),
        )
    ).view(d_model, d_model)
    full_grad[-1] = -full_grad[:-1].sum(dim=0)
    return full_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
class ProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for two attention-output shift quotients."""
=======
class ProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for all attention-output shift quotients."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        d_model = self.module.d_model
        missing_start = self.module.missing_start
        for group in self.param_groups:
=======
        d_model = self.module.d_model
        for group in self.param_groups:
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full_value[-1, :2].clone()
                full_value[:, :2].sub_(omitted)
                full_value[-1, :2].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 2 :],
                        )
                    )
                )
=======
                omitted = full_value[-1].clone()
                full_value.sub_(omitted)
                full_value[-1].zero_()
                parameter.copy_(full_value[:-1].reshape(-1))
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_start = d_model * d_ff - 11
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(11),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -11:] = -full_grad[:-1, -11:].sum(dim=0)
    return full_grad
=======
def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    full_grad = torch.cat(
        (
            parameter.grad,
            parameter.grad.new_zeros(d_ff),
        )
    ).view(d_model, d_ff)
    full_grad[-1] = -full_grad[:-1].sum(dim=0)
    return full_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for eleven MLP common-output shift quotients."""
=======
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for all MLP common-output shift quotients."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        d_model = self.module.d_model
        d_ff = self.module.d_ff
        missing_start = self.module.missing_start
        for group in self.param_groups:
=======
        d_model = self.module.d_model
        d_ff = self.module.d_ff
        for group in self.param_groups:
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full_value[-1, -11:].clone()
                full_value[:, -11:].sub_(omitted)
                full_value[-1, -11:].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 11 :],
                        )
                    )
                )
=======
                omitted = full_value[-1].clone()
                full_value.sub_(omitted)
                full_value[-1].zero_()
                parameter.copy_(full_value[:-1].reshape(-1))
>>>>>>> REPLACE

<<<<<<< SEARCH
                total_sq.add_(
                    full_grad[-1, :2].float().square().sum()
                )
=======
                total_sq.add_(
                    full_grad[-1].float().square().sum()
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                total_sq.add_(
                    full_grad[-1, -11:].float().square().sum()
                )
=======
                total_sq.add_(
                    full_grad[-1].float().square().sum()
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
    output_bias_gauge_parameters = [
        block.mlp.fc2.bias for block in model.blocks
    ]
=======
    embedding_gauge_parameters = [
        model.token_emb.weight
    ]
    output_bias_gauge_parameters = [
        block.mlp.fc2.bias for block in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    last_coordinate_gauge_parameters = [
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
    ]
=======
    last_coordinate_gauge_parameters = [
        *embedding_gauge_parameters,
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
    ]
>>>>>>> REPLACE