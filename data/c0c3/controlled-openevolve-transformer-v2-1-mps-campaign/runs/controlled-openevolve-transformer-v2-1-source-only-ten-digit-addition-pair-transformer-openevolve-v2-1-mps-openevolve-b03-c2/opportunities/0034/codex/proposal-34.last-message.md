MECHANISM: Trajectory-preserving LayerNorm-scale folding into the MLP input projection

HYPOTHESIS: Starting from the qualified 1587-parameter complete value-bias quotient, folding all eight `ln2` scales into `fc1.weight` while reproducing their full AdamW and clipping dynamics will yield a 1579-parameter model with at least 99% accuracy.

INTENDED_EDIT: Remove the complete attention value bias and all `ln2` affine parameters, then train the effective `fc1.weight` through optimizer-maintained weight/scale factors whose product is stored in the model.

EVIDENCE: Complete value-bias absorption achieved 99.94% at 1587 parameters. Fixing one `ln2` scale without preserving its optimization trajectory collapsed to 28.96%, motivating an exact folded-factor update instead of a fixed-scale optimization change.

<<<<<<< SEARCH
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain all but two value-bias coordinates; the omitted values
        # are absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 2))
=======
        # Key bias is softmax-invariant, and the complete value bias can be
        # absorbed by the downstream projection bias. Store only query bias.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = self.qkv.bias[:d_model]
        value_bias = torch.cat(
            (self.qkv.bias[d_model:], self.qkv.bias.new_zeros(2))
        )
=======
        query_bias = self.qkv.bias
        value_bias = self.qkv.bias.new_zeros(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.bias = None
=======
        # Its learned scale is folded into fc1.weight by the optimizer.
        self.ln2 = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
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
            value_bias_grads.append(grad)

        saved_grads = [param.grad for param in self.quotient_params]
        for param in self.quotient_params:
            param.grad = None

        loss = super().step(closure)
=======
            value_bias_grads.append(grad)

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
            param.add_(
                full_update[..., :-1] - full_update[..., -1:],
                alpha=-step_size,
            )

        # Update the omitted value-bias coordinate in full-coordinate AdamW,
=======
            param.add_(
                full_update[..., :-1] - full_update[..., -1:],
                alpha=-step_size,
            )

        # Reproduce AdamW on the full fc1 weight and the omitted ln2 scale,
        # then store their functionally sufficient columnwise product.
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

        # Update the omitted value-bias coordinate in full-coordinate AdamW,
>>>>>>> REPLACE

<<<<<<< SEARCH
def clip_rowwise_quotient_grad_norm_(
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
=======
def clip_rowwise_quotient_grad_norm_(
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    value_bias_specs = [
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj.bias,
        )
        for block in model.blocks
    ]
    optimizer = RowwiseQuotientAdamW(
        model.parameters(),
        quotient_params,
        value_bias_specs,
        lr=train_cfg.lr,
=======
    value_bias_specs = [
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj.bias,
        )
        for block in model.blocks
    ]
    factor_params = [
        block.mlp.fc1.weight for block in model.blocks
    ]
    optimizer = RowwiseQuotientAdamW(
        model.parameters(),
        quotient_params,
        value_bias_specs,
        factor_params,
        lr=train_cfg.lr,
>>>>>>> REPLACE

<<<<<<< SEARCH
                value_bias_specs,
                train_cfg.grad_clip,
            )
=======
                value_bias_specs,
                train_cfg.grad_clip,
                optimizer,
            )
>>>>>>> REPLACE