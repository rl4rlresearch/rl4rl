MECHANISM: Content-independent dual-head learned lag routing

HYPOTHESIS: Replacing token-dependent query/key dot products with two learned causal relative-lag attention distributions will retain at least 99% accuracy with 1,338 parameters, because the fixed-format task needs head-specific operand routes but may not require those routes to vary with token content.

INTENDED_EDIT: Start from the qualified nine-column MLP gauge, replace QKV with a learned value-only projection, route attention solely through the existing head-specific lag tables, and gauge-fix the now-unshared attention output bias.

EVIDENCE: Learned relative-lag routing achieved 99.96% at 1,476 parameters, while reducing token-representation rank collapsed to 5.06%; this motivates preserving full-rank token embeddings and testing the distinct assumption that content-dependent Q/K routing—not representation capacity—is expendable.

<<<<<<< SEARCH
class GaugeFixedKeyLinear(nn.Linear):
    """QKV projection with one key-row coefficient fixed."""

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
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with three common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 3
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 3))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(3),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)
=======
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with nine common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 9
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 9))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(9),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 7))
        self.proj = GaugeFixedProjectionLinear(d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.value = FixedRouteValueLinear(d_model)
        self.proj = GaugeFixedProjectionLinear(d_model)
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
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
        positions = torch.arange(seqlen, device=x.device)
=======
        bsz, seqlen, d_model = x.shape
        v = self.value(x)
        v = v.view(
            bsz, seqlen, self.n_head, self.head_dim
        ).transpose(1, 2)

        positions = torch.arange(seqlen, device=x.device)
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
        if isinstance(module, GaugeFixedKeyLinear):
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
        if isinstance(module, FixedRouteValueLinear):
            d_model = module.d_model
            full = module.weight.new_empty(3 * d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full[2 * d_model :])
        elif isinstance(module, GaugeFixedProjectionLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full[-1, -3:].clone()
                full[:, -3:].sub_(omitted)
                full[-1, -3:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 3 :],
                        )
                    )
                )
=======
                omitted = full[-1, -9:].clone()
                full[:, -9:].sub_(omitted)
                full[-1, -9:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 9 :],
                        )
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_start = d_model * d_ff - 3
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(3),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -3:] = -full_grad[:-1, -3:].sum(dim=0)
    return full_grad
=======
def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_start = d_model * d_ff - 9
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(9),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -9:] = -full_grad[:-1, -9:].sum(dim=0)
    return full_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for three MLP common-output shift quotients."""
=======
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for nine MLP common-output shift quotients."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full_value[-1, -3:].clone()
                full_value[:, -3:].sub_(omitted)
                full_value[-1, -3:].zero_()
                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 3 :],
                        )
                    )
                )
=======
                omitted = full_value[-1, -9:].clone()
                full_value[:, -9:].sub_(omitted)
                full_value[-1, -9:].zero_()
                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 9 :],
                        )
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                total_sq.add_(
                    full_grad[-1, -3:].float().square().sum()
                )
=======
                total_sq.add_(
                    full_grad[-1, -9:].float().square().sum()
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
=======
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
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
    ]
    key_gauge_modules = []
>>>>>>> REPLACE

<<<<<<< SEARCH
    shared_query_projection_parameters = [
        block.attn.proj.bias for block in model.blocks
    ]
=======
    shared_query_projection_parameters = []
>>>>>>> REPLACE