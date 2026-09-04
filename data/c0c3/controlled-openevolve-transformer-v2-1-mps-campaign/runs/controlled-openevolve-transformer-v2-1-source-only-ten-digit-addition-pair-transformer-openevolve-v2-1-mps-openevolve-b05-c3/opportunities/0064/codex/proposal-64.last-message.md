MECHANISM: Global tied-embedding common-shift quotient

HYPOTHESIS: Applying an exact one-coordinate common-shift gauge to Reference Design 2’s tied token embedding/head will reduce the model from 1,335 to 1,334 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Adopt the qualified value-only learned-lag attention and complete twelve-column MLP gauge, then represent the tied embedding modulo one global scalar while preserving full-space initialization, AdamW moments, weight decay, and gradient clipping.

EVIDENCE: Reference Design 2 achieved 99.87% accuracy with 1,335 parameters; subtracting one scalar from every embedding entry changes residual inputs only by LayerNorm-invariant channel shifts and changes every output logit by the same softmax-invariant scalar.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Embedding):
    """Embedding vectors represented modulo a shared channel shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the constructor RNG used by the original full embedding.
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
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedKeyLinear(nn.Linear):
    """QKV projection with one key-row coefficient fixed by LayerNorm gauge."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.missing_index = d_model * d_model + d_model - 1
        self.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 1))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_index],
                self.weight.new_zeros(1),
                self.weight[self.missing_index :],
            )
        )
        return flat.view(3 * self.d_model, self.d_model)
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
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with seven common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 7
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 7))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(7),
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
        self.n_head = n_head
        self.head_dim = d_model // n_head
        # One key-row coefficient is fixed by the LayerNorm gauge. One query
        # coordinate is explicit, a second uses the projection-bias shift, and
        # their mean supplies the third active query coordinate.
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 7))
        self.proj = GaugeFixedProjectionLinear(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.relative_bias = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.value = FixedRouteValueLinear(d_model)
        self.proj = GaugeFixedProjectionLinear(d_model)
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.relative_bias = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        independent_query_bias = torch.cat(
            (self.qkv.bias, self.proj.bias[-1:])
        )
        query_bias = torch.cat(
            (
                independent_query_bias,
                independent_query_bias.mean().unsqueeze(0),
            )
        )
        bias = torch.cat(
            (query_bias, self.qkv.bias.new_zeros(2 * d_model + 5))
        )
        qkv = F.linear(x, self.qkv.full_weight(), bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
=======
        bsz, seqlen, d_model = x.shape
        v = self.value(x)
        v = v.view(
            bsz, seqlen, self.n_head, self.head_dim
        ).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = att + lag_bias[:, lag].unsqueeze(0)
=======
        att = lag_bias[:, lag].unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = F.linear(y, self.proj.full_weight(), self.proj.bias)
=======
        output_bias = torch.cat(
            (self.proj.bias, self.proj.bias.new_zeros(1))
        )
        y = F.linear(y, self.proj.full_weight(), output_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedEmbedding):
            # Draw the original full tensor to preserve the qualified RNG
            # stream, then select the equivalent last-coordinate-zero gauge.
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
        elif isinstance(module, GaugeFixedKeyLinear):
            d_model = module.d_model
            full = module.weight.new_empty(3 * d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[d_model, -1].clone()
                full[d_model, :-1].sub_(omitted)
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_index],
                            flat[module.missing_index + 1 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
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
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
            full = module.weight.new_empty(
                module.d_model, module.d_ff
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, -7:].clone()
                full[:, -7:].sub_(omitted)
                full[-1, -7:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 7 :],
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
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
=======
        self.token_emb = GaugeFixedEmbedding(
            cfg.vocab_size, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        # Weight tying with the reconstructed full input embedding.
        self.lm_head = GaugeTiedHead(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_start = d_model * d_ff - 7
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(7),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -7:] = -full_grad[:-1, -7:].sum(dim=0)
    return full_grad
=======
def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_start = (d_model - 1) * d_ff
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(d_ff),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1] = -full_grad[:-1].sum(dim=0)
    return full_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for seven MLP common-output shift quotients."""
=======
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for all MLP common-output shift quotients."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full_value[-1, -7:].clone()
                full_value[:, -7:].sub_(omitted)
                full_value[-1, -7:].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 7 :],
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
                    full_grad[-1, -7:].float().square().sum()
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
    relative_bias_gauge_parameters = [
        block.attn.relative_bias for block in model.blocks
    ]
    last_coordinate_gauge_parameters = [
        *output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
    ]
    key_gauge_modules = [
        (block.attn.qkv, block.ln1.weight)
        for block in model.blocks
    ]
    projection_gauge_modules = [
        block.attn.proj for block in model.blocks
    ]
    mlp_projection_gauge_modules = [
        block.mlp.fc2 for block in model.blocks
    ]
    shared_query_projection_parameters = [
        block.attn.proj.bias for block in model.blocks
    ]
=======
    embedding_gauge_parameters = [
        model.token_emb.weight
    ]
    output_bias_gauge_parameters = [
        block.mlp.fc2.bias for block in model.blocks
    ]
    attention_output_bias_gauge_parameters = [
        block.attn.proj.bias for block in model.blocks
    ]
    relative_bias_gauge_parameters = [
        block.attn.relative_bias for block in model.blocks
    ]
    last_coordinate_gauge_parameters = [
        *embedding_gauge_parameters,
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
    ]
    key_gauge_modules = []
    projection_gauge_modules = [
        block.attn.proj for block in model.blocks
    ]
    mlp_projection_gauge_modules = [
        block.mlp.fc2 for block in model.blocks
    ]
    shared_query_projection_parameters = []
>>>>>>> REPLACE