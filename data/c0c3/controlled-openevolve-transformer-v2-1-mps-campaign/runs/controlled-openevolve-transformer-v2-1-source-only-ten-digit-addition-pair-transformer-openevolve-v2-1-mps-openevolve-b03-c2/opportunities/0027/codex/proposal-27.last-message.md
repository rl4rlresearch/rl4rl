MECHANISM: Three-coordinate value-bias absorption into the attention projection bias

HYPOTHESIS: Extending the verified two-coordinate value-bias quotient to three omitted coordinates will produce a 1592-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the newly omitted coordinate.

INTENDED_EDIT: Store five learned value-bias coordinates, reconstruct three zero-gauge coordinates, and absorb their independently tracked AdamW updates into `attn.proj.bias`.

EVIDENCE: Omitting one value-bias coordinate achieved 99.73% at 1594 parameters, and omitting two achieved 99.98% at 1593; the third coordinate has the same attention-invariant computational role.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Reuse the mean learned query bias across every softmax-invariant
        # key-bias coordinate, leaving only query and value bias parameters.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain all but three value-bias coordinates; the omitted values
        # are absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 3))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = self.qkv.bias[:d_model]
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                self.qkv.bias[d_model:],
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
=======
        query_bias = self.qkv.bias[:d_model]
        value_bias = torch.cat(
            (self.qkv.bias[d_model:], self.qkv.bias.new_zeros(3))
        )
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                value_bias,
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
class QuotientAdamW(torch.optim.AdamW):
    """AdamW preserving omitted uniform coordinates along the last axis."""

    def __init__(self, params, quotient_params, **kwargs):
        self.quotient_params = list(quotient_params)
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        saved_grads = [param.grad for param in self.quotient_params]
=======
class QuotientAdamW(torch.optim.AdamW):
    """AdamW preserving omitted uniform and absorbed-bias coordinates."""

    def __init__(
        self, params, quotient_params, value_bias_specs=(), **kwargs
    ):
        self.quotient_params = list(quotient_params)
        self.value_bias_specs = list(value_bias_specs)
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        value_bias_grads = []
        for qkv_bias, proj_weight, proj_bias in self.value_bias_specs:
            grad = None
            if proj_bias.grad is not None:
                omitted_dims = (
                    2 * proj_weight.size(1) - qkv_bias.numel()
                )
                grad = (
                    proj_weight.detach()[:, -omitted_dims:]
                    * proj_bias.grad.detach().unsqueeze(1)
                ).sum(dim=0)
            value_bias_grads.append(grad)

        saved_grads = [param.grad for param in self.quotient_params]
>>>>>>> REPLACE

<<<<<<< SEARCH
            param.add_(
                full_update[..., :-1] - full_update[..., -1:],
                alpha=-step_size,
            )

        return loss
=======
            param.add_(
                full_update[..., :-1] - full_update[..., -1:],
                alpha=-step_size,
            )

        # Track each omitted value-bias coordinate with full-coordinate
        # AdamW, then return to the zero gauge by folding its update into
        # the already-updated attention projection bias.
        for (qkv_bias, proj_weight, proj_bias), grad in zip(
            self.value_bias_specs, value_bias_grads
        ):
            if grad is None:
                continue

            group = next(
                group
                for group in self.param_groups
                if any(candidate is qkv_bias for candidate in group["params"])
            )
            if group["maximize"]:
                grad = -grad

            state = self.state[qkv_bias]
            if "value_quotient_step" not in state:
                state["value_quotient_step"] = 0
                state["value_quotient_exp_avg"] = qkv_bias.new_zeros(
                    grad.shape
                )
                state["value_quotient_exp_avg_sq"] = qkv_bias.new_zeros(
                    grad.shape
                )

            state["value_quotient_step"] += 1
            step = state["value_quotient_step"]
            beta1, beta2 = group["betas"]
            exp_avg = state["value_quotient_exp_avg"]
            exp_avg_sq = state["value_quotient_exp_avg_sq"]

            exp_avg.lerp_(grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                grad, grad, value=1.0 - beta2
            )

            step_size = group["lr"] / (1.0 - beta1 ** step)
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(1.0 - beta2 ** step)
            ).add_(group["eps"])
            omitted_value = -step_size * exp_avg / denom
            proj_bias.add_(
                proj_weight[:, -omitted_value.numel():] @ omitted_value
            )

        return loss
>>>>>>> REPLACE

<<<<<<< SEARCH
def clip_quotient_grad_norm_(parameters, quotient_params, max_norm: float) -> None:
=======
def clip_quotient_grad_norm_(
    parameters, quotient_params, value_bias_specs, max_norm: float
) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
        total_sq = term if total_sq is None else total_sq + term

    if total_sq is None:
=======
        total_sq = term if total_sq is None else total_sq + term

    for qkv_bias, proj_weight, proj_bias in value_bias_specs:
        if proj_bias.grad is None:
            continue
        omitted_dims = 2 * proj_weight.size(1) - qkv_bias.numel()
        omitted_grad = (
            proj_weight.detach()[:, -omitted_dims:]
            * proj_bias.grad.detach().unsqueeze(1)
        ).sum(dim=0)
        term = omitted_grad.square().sum()
        total_sq = term if total_sq is None else total_sq + term

    if total_sq is None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    quotient_params = [model.token_emb.weight, model.pos_emb.weight] + [
        block.mlp.fc2.bias for block in model.blocks
    ]
    optimizer = QuotientAdamW(
        model.parameters(),
        quotient_params,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
=======
    quotient_params = [model.token_emb.weight, model.pos_emb.weight] + [
        block.mlp.fc2.bias for block in model.blocks
    ]
    value_bias_specs = [
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj.bias,
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            clip_quotient_grad_norm_(
                model.parameters(), quotient_params, train_cfg.grad_clip
            )
=======
            clip_quotient_grad_norm_(
                model.parameters(),
                quotient_params,
                value_bias_specs,
                train_cfg.grad_clip,
            )
>>>>>>> REPLACE