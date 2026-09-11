MECHANISM: MLP residual-output matrix quotient with trajectory-preserving LayerNorm folding

HYPOTHESIS: Extending the qualified 1570-parameter design by removing the feature-uniform output coordinate from each of the 12 `fc2.weight` columns will produce a 1558-parameter model with at least 99% accuracy, because those components add only tokenwise uniform residual shifts canceled by the final LayerNorm.

INTENDED_EDIT: Adopt the qualified complete value-bias absorption, folded `ln1`/`ln2` scales, and relative attention-projection bias; additionally store `fc2.weight` as twelve rows of seven relative output coordinates and preserve its full-coordinate AdamW and clipping trajectory.

EVIDENCE: Reference Design 3 achieved 99.98% accuracy at 1570 parameters after removing the uniform coordinate of both residual-output biases. The same invariance applies independently to every `fc2.weight` column, since each omitted coordinate contributes only a feature-uniform residual component before the final LayerNorm.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain only the second value-bias coordinate; the other seven
        # are absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 7))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, and every value-bias coordinate can
        # be absorbed by the downstream projection bias. Store only query
        # bias and reconstruct the other two bias vectors in fixed gauges.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
        # Downstream LayerNorms cancel its feature-uniform coordinate.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = self.qkv.bias[:d_model]
        retained_value_bias = self.qkv.bias[d_model:]
        value_bias = torch.cat(
            (
                retained_value_bias.new_zeros(1),
                retained_value_bias,
                retained_value_bias.new_zeros(d_model - 2),
            )
        )
=======
        query_bias = self.qkv.bias
        value_bias = self.qkv.bias.new_zeros(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        relative_bias = torch.cat(
            (self.proj.bias, self.proj.bias.new_zeros(1))
        )
        proj_bias = relative_bias + self.proj.bias.mean()
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        relative_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        fc2_bias = relative_bias + self.fc2.bias.mean()
        return self.drop(F.linear(hidden, self.fc2.weight, fc2_bias))
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        weight_relative = torch.cat(
            (
                self.fc2.weight,
                self.fc2.weight.new_zeros(
                    (self.fc2.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        fc2_weight = (
            weight_relative
            + self.fc2.weight.mean(dim=-1, keepdim=True)
        ).transpose(0, 1)
        relative_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        fc2_bias = relative_bias + self.fc2.bias.mean()
        return self.drop(F.linear(hidden, fc2_weight, fc2_bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln1.bias = None
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.bias = None
=======
        # Learned scales are folded into their downstream weights by the
        # factor-aware optimizer.
        self.ln1 = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # A global feature-uniform shift of every tied token row is canceled
=======
        self.apply(self._init_weights)

        # Each MLP-output column is defined only modulo a feature-uniform
        # vector, which the final LayerNorm cancels. Store relative output
        # coordinates with the hidden dimension first.
        for block in self.blocks:
            full_fc2_weight = block.mlp.fc2.weight.detach().transpose(0, 1)
            block.mlp.fc2.weight = nn.Parameter(
                full_fc2_weight[:, :-1] - full_fc2_weight[:, -1:]
            )

        # A global feature-uniform shift of every tied token row is canceled
>>>>>>> REPLACE

<<<<<<< SEARCH
    def __init__(
        self, params, quotient_params, value_bias_specs=(), **kwargs
    ):
        self.quotient_params = list(quotient_params)
        self.value_bias_specs = list(value_bias_specs)
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
=======
    def __init__(
        self,
        params,
        quotient_params,
        value_bias_specs=(),
        factor_params=(),
        **kwargs,
    ):
        self.quotient_params = list(quotient_params)
        self.value_bias_specs = list(value_bias_specs)
        self.factor_params = list(factor_params)
        super().__init__(params, **kwargs)

    def _factor_state(self, param):
        state = self.state[param]
        if "factor_weight" not in state:
            state["factor_step"] = 0
            state["factor_weight"] = param.detach().clone()
            state["factor_scale"] = param.new_ones(param.size(1))
            state["factor_weight_exp_avg"] = torch.zeros_like(param)
            state["factor_weight_exp_avg_sq"] = torch.zeros_like(param)
            state["factor_scale_exp_avg"] = param.new_zeros(
                param.size(1)
            )
            state["factor_scale_exp_avg_sq"] = param.new_zeros(
                param.size(1)
            )
        return state

    @torch.no_grad()
    def factor_grad_sq(self, param):
        state = self._factor_state(param)
        grad = param.grad.detach()
        weight_grad = grad * state["factor_scale"].unsqueeze(0)
        scale_grad = (
            grad * state["factor_weight"]
        ).sum(dim=0)
        return weight_grad.square().sum() + scale_grad.square().sum()

    @torch.no_grad()
    def step(self, closure=None):
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias_grads = []
        for (
            qkv_bias,
            proj_weight,
            proj_bias,
            retained_dims,
        ) in self.value_bias_specs:
            grad = None
            if proj_bias.grad is not None:
                omitted_dims = [
                    dim
                    for dim in range(proj_weight.size(1))
                    if dim not in retained_dims
                ]
                grad = (
                    proj_weight.detach()[:, omitted_dims]
                    * proj_bias.grad.detach().unsqueeze(1)
                ).sum(dim=0)
            value_bias_grads.append(grad)
=======
        value_bias_grads = []
        for qkv_bias, proj_weight, proj_bias in self.value_bias_specs:
            grad = None
            if proj_bias.grad is not None:
                full_proj_grad = torch.cat(
                    (
                        proj_bias.grad.detach(),
                        -proj_bias.grad.detach().sum().view(1),
                    )
                )
                grad = (
                    proj_weight.detach()
                    * full_proj_grad.unsqueeze(1)
                ).sum(dim=0)
            value_bias_grads.append(grad)
>>>>>>> REPLACE

<<<<<<< SEARCH
        saved_grads = [param.grad for param in self.quotient_params]
        for param in self.quotient_params:
            param.grad = None

        loss = super().step(closure)
=======
        saved_grads = [param.grad for param in self.quotient_params]
        saved_factor_grads = [
            param.grad for param in self.factor_params
        ]
        for param in self.quotient_params:
            param.grad = None
        for param in self.factor_params:
            param.grad = None

        loss = super().step(closure)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Track each omitted value-bias coordinate with full-coordinate
        # AdamW, then return to the zero gauge by folding its update into
        # the already-updated attention projection bias.
=======
        # Reproduce AdamW on each downstream weight and its omitted
        # LayerNorm scale, then store their sufficient columnwise product.
        for param, grad in zip(
            self.factor_params, saved_factor_grads
        ):
            param.grad = grad
            if grad is None:
                continue

            group = next(
                group
                for group in self.param_groups
                if any(candidate is param for candidate in group["params"])
            )
            state = self._factor_state(param)
            factor_weight = state["factor_weight"]
            factor_scale = state["factor_scale"]
            weight_grad = grad * factor_scale.unsqueeze(0)
            scale_grad = (grad * factor_weight).sum(dim=0)
            if group["maximize"]:
                weight_grad = -weight_grad
                scale_grad = -scale_grad

            state["factor_step"] += 1
            step = state["factor_step"]
            beta1, beta2 = group["betas"]
            weight_exp_avg = state["factor_weight_exp_avg"]
            weight_exp_avg_sq = state["factor_weight_exp_avg_sq"]
            scale_exp_avg = state["factor_scale_exp_avg"]
            scale_exp_avg_sq = state["factor_scale_exp_avg_sq"]

            weight_exp_avg.lerp_(weight_grad, 1.0 - beta1)
            weight_exp_avg_sq.mul_(beta2).addcmul_(
                weight_grad, weight_grad, value=1.0 - beta2
            )
            scale_exp_avg.lerp_(scale_grad, 1.0 - beta1)
            scale_exp_avg_sq.mul_(beta2).addcmul_(
                scale_grad, scale_grad, value=1.0 - beta2
            )

            lr = group["lr"]
            decay = 1.0 - lr * group["weight_decay"]
            factor_weight.mul_(decay)
            factor_scale.mul_(decay)
            step_size = lr / (1.0 - beta1 ** step)
            bias_correction2 = math.sqrt(1.0 - beta2 ** step)
            weight_denom = weight_exp_avg_sq.sqrt().div_(
                bias_correction2
            ).add_(group["eps"])
            scale_denom = scale_exp_avg_sq.sqrt().div_(
                bias_correction2
            ).add_(group["eps"])
            factor_weight.addcdiv_(
                weight_exp_avg,
                weight_denom,
                value=-step_size,
            )
            factor_scale.addcdiv_(
                scale_exp_avg,
                scale_denom,
                value=-step_size,
            )
            param.copy_(
                factor_weight * factor_scale.unsqueeze(0)
            )

        # Track each omitted value-bias coordinate with full-coordinate
        # AdamW, then return to the zero gauge by folding its update into
        # the already-updated attention projection bias.
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (
            qkv_bias,
            proj_weight,
            proj_bias,
            retained_dims,
        ), grad in zip(self.value_bias_specs, value_bias_grads):
=======
        for (qkv_bias, proj_weight, proj_bias), grad in zip(
            self.value_bias_specs, value_bias_grads
        ):
>>>>>>> REPLACE

<<<<<<< SEARCH
            omitted_value = -step_size * exp_avg / denom
            omitted_dims = [
                dim
                for dim in range(proj_weight.size(1))
                if dim not in retained_dims
            ]
            proj_bias.add_(
                proj_weight[:, omitted_dims] @ omitted_value
            )
=======
            omitted_value = -step_size * exp_avg / denom
            full_delta = proj_weight @ omitted_value
            proj_bias.add_(
                full_delta[:-1] - full_delta[-1]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
@torch.no_grad()
def clip_quotient_grad_norm_(
    parameters, quotient_params, value_bias_specs, max_norm: float
) -> None:
    parameters = list(parameters)
    quotient_ids = {id(param) for param in quotient_params}
    total_sq = None

    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        term = grad.square().sum()
        if id(param) in quotient_ids:
            term = term + grad.sum(dim=-1).square().sum()
        total_sq = term if total_sq is None else total_sq + term

    for qkv_bias, proj_weight, proj_bias, retained_dims in value_bias_specs:
        if proj_bias.grad is None:
            continue
        omitted_dims = [
            dim
            for dim in range(proj_weight.size(1))
            if dim not in retained_dims
        ]
        omitted_grad = (
            proj_weight.detach()[:, omitted_dims]
            * proj_bias.grad.detach().unsqueeze(1)
        ).sum(dim=0)
        term = omitted_grad.square().sum()
        total_sq = term if total_sq is None else total_sq + term

    if total_sq is None:
        return

    clip_coef = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for param in parameters:
        if param.grad is not None:
            param.grad.mul_(clip_coef)
=======
@torch.no_grad()
def clip_quotient_grad_norm_(
    parameters,
    quotient_params,
    value_bias_specs,
    max_norm: float,
    factor_optimizer=None,
) -> None:
    parameters = list(parameters)
    quotient_ids = {id(param) for param in quotient_params}
    factor_ids = (
        {id(param) for param in factor_optimizer.factor_params}
        if factor_optimizer is not None
        else set()
    )
    total_sq = None

    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if id(param) in factor_ids:
            term = factor_optimizer.factor_grad_sq(param)
        else:
            term = grad.square().sum()
            if id(param) in quotient_ids:
                term = term + grad.sum(dim=-1).square().sum()
        total_sq = term if total_sq is None else total_sq + term

    for qkv_bias, proj_weight, proj_bias in value_bias_specs:
        if proj_bias.grad is None:
            continue
        full_proj_grad = torch.cat(
            (
                proj_bias.grad.detach(),
                -proj_bias.grad.detach().sum().view(1),
            )
        )
        omitted_grad = (
            proj_weight.detach()
            * full_proj_grad.unsqueeze(1)
        ).sum(dim=0)
        term = omitted_grad.square().sum()
        total_sq = term if total_sq is None else total_sq + term

    if total_sq is None:
        return

    clip_coef = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for param in parameters:
        if param.grad is not None:
            param.grad.mul_(clip_coef)
>>>>>>> REPLACE

<<<<<<< SEARCH
    quotient_params = [model.token_emb.weight, model.pos_emb.weight] + [
        block.mlp.fc2.bias for block in model.blocks
    ]
    value_bias_specs = [
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj.bias,
            (1,),
        )
        for block in model.blocks
    ]
    optimizer = QuotientAdamW(
        model.parameters(),
        quotient_params,
        value_bias_specs,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
=======
    quotient_params = [model.token_emb.weight, model.pos_emb.weight] + [
        block.attn.proj.bias for block in model.blocks
    ] + [
        block.mlp.fc2.bias for block in model.blocks
    ] + [
        block.mlp.fc2.weight for block in model.blocks
    ]
    value_bias_specs = [
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj.bias,
        )
        for block in model.blocks
    ]
    factor_params = [
        block.attn.qkv.weight for block in model.blocks
    ] + [
        block.mlp.fc1.weight for block in model.blocks
    ]
    optimizer = QuotientAdamW(
        model.parameters(),
        quotient_params,
        value_bias_specs,
        factor_params,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
            clip_quotient_grad_norm_(
                model.parameters(),
                quotient_params,
                value_bias_specs,
                train_cfg.grad_clip,
            )
=======
            clip_quotient_grad_norm_(
                model.parameters(),
                quotient_params,
                value_bias_specs,
                train_cfg.grad_clip,
                optimizer,
            )
>>>>>>> REPLACE