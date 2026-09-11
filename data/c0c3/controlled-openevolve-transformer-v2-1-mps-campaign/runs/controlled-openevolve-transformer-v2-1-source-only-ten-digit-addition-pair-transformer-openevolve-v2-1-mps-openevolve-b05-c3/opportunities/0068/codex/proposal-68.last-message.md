MECHANISM: Single-coordinate pre-attention LayerNorm-scale quotient

HYPOTHESIS: Starting from the verified 1,320-parameter design, fixing one `ln1` scale coordinate at one will produce 1,319 parameters and retain at least 99% accuracy, because the learned value projection can absorb that channel scale while the other seven adaptive scales preserve the optimization flexibility lost by the failed fully non-affine design.

INTENDED_EDIT: Adopt the verified global embedding, complete attention/MLP projection, and `ln1`-bias gauges, then quotient only the final `ln1` scale coordinate while preserving full-shape initialization and gauge-aware optimization elsewhere.

EVIDENCE: The affine-without-bias design achieved 99.66% at 1,320 parameters, whereas removing all eight remaining `ln1` scales collapsed to 42.51%; a one-coordinate quotient directly tests whether seven adaptive scales suffice without repeating the destructive all-at-once reduction.

<<<<<<< SEARCH
    vocab_size: int


class FixedRouteValueLinear(nn.Linear):
=======
    vocab_size: int


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
    """LayerNorm with one scale absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(1)))

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
        self.weight = nn.Parameter(torch.empty(d_model, d_model))
        self.bias = None


class GaugeFixedProjectionLinear(nn.Linear):
=======
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
    """Attention projection with two common-output shifts fixed."""

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
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with ten common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 10
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 10))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(10),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)
=======
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
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        self.cfg = cfg
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
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, FixedRouteValueLinear):
            d_model = module.d_model
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedEmbedding):
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                full.sub_(full[-1, -1].clone())
                module.weight.copy_(full.reshape(-1)[:-1])
        elif isinstance(module, FixedRouteValueLinear):
            d_model = module.d_model
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedProjectionLinear):
            d_model = module.d_model
=======
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
            d_model = module.d_model
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
=======
            with torch.no_grad():
                omitted = full[-1].clone()
                full.sub_(omitted)
                full[-1].zero_()
                module.weight.copy_(full[:-1].reshape(-1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
            with torch.no_grad():
                omitted = full[-1, -10:].clone()
                full[:, -10:].sub_(omitted)
                full[-1, -10:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 10 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
=======
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
    missing_start = d_model * d_ff - 10
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(10),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -10:] = -full_grad[:-1, -10:].sum(dim=0)
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
    """Full-space AdamW for ten MLP common-output shift quotients."""
=======
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for all MLP common-output shift quotients."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full_value[-1, -10:].clone()
                full_value[:, -10:].sub_(omitted)
                full_value[-1, -10:].zero_()
                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 10 :],
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
                    full_grad[-1, -10:].float().square().sum()
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