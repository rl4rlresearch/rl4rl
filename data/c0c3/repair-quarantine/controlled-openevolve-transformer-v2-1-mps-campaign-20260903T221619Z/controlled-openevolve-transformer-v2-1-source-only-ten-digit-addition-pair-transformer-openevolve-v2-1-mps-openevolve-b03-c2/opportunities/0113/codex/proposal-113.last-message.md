MECHANISM: Projected duplicated-moment AdamW for tied terminal gains

HYPOTHESIS: Tying the last two learned final-LayerNorm gains while optimizing their coordinate gradients with separate AdamW moments will reduce the model from 1485 to 1484 parameters and achieve at least 99% accuracy.

INTENDED_EDIT: Replace the final two learned gain coordinates with one shared parameter, capture their individual gradients, and project separate full-coordinate AdamW updates back onto the shared gain.

EVIDENCE: Naive adjacent final-LayerNorm gain sharing reached 98.91%, the closest verified 1484 result, while the current unrestricted design reached 99.92%; preserving separate adaptive moments directly tests whether the narrow deficit was caused by the tied parameter’s aggregated optimizer dynamics.

<<<<<<< SEARCH
        self.ln_f_weight = nn.Parameter(torch.ones(cfg.d_model - 1))
        self.ln_f_bias = nn.Parameter(torch.zeros(cfg.d_model))
=======
        self.ln_f_weight = nn.Parameter(torch.ones(cfg.d_model - 2))
        self.ln_f_bias = nn.Parameter(torch.zeros(cfg.d_model))
        self._ln_f_full_grad = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        ln_f_weight = torch.cat(
            (
                self.ln_f_weight,
                self.ln_f_weight.new_ones(1),
            )
        )
        x = x * ln_f_weight + self.ln_f_bias
=======
        ln_f_weight = torch.cat(
            (
                self.ln_f_weight[:-1],
                self.ln_f_weight[-1:].expand(2),
                self.ln_f_weight.new_ones(1),
            )
        )
        if self.training and torch.is_grad_enabled():
            self._ln_f_full_grad = None
            ln_f_weight.register_hook(
                lambda grad: setattr(
                    self, "_ln_f_full_grad", grad.detach()
                )
            )
        x = x * ln_f_weight + self.ln_f_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias_specs=(),
        factor_params=(),
        embedding_specs=(),
        **kwargs,
    ):
        self.quotient_params = list(quotient_params)
        self.value_bias_specs = list(value_bias_specs)
        self.factor_params = list(factor_params)
        self.embedding_specs = list(embedding_specs)
        super().__init__(params, **kwargs)
=======
        value_bias_specs=(),
        factor_params=(),
        embedding_specs=(),
        tied_gain_specs=(),
        **kwargs,
    ):
        self.quotient_params = list(quotient_params)
        self.value_bias_specs = list(value_bias_specs)
        self.factor_params = list(factor_params)
        self.embedding_specs = list(embedding_specs)
        self.tied_gain_specs = list(tied_gain_specs)
        super().__init__(params, **kwargs)
>>>>>>> REPLACE

<<<<<<< SEARCH
        saved_embedding_grads = [
            (token_param.grad, pos_param.grad)
            for token_param, pos_param in self.embedding_specs
        ]
        for param in self.quotient_params:
            param.grad = None
        for param in self.factor_params:
            param.grad = None
        for token_param, pos_param in self.embedding_specs:
            token_param.grad = None
            pos_param.grad = None

        loss = super().step(closure)

        for param, grad in zip(self.quotient_params, saved_grads):
=======
        saved_embedding_grads = [
            (token_param.grad, pos_param.grad)
            for token_param, pos_param in self.embedding_specs
        ]
        saved_tied_gain_grads = [
            (param.grad, owner._ln_f_full_grad)
            for param, owner in self.tied_gain_specs
        ]
        for param in self.quotient_params:
            param.grad = None
        for param in self.factor_params:
            param.grad = None
        for token_param, pos_param in self.embedding_specs:
            token_param.grad = None
            pos_param.grad = None
        for param, _ in self.tied_gain_specs:
            param.grad = None

        loss = super().step(closure)

        for (
            param,
            _,
        ), (
            aggregated_grad,
            full_grad,
        ) in zip(self.tied_gain_specs, saved_tied_gain_grads):
            param.grad = aggregated_grad
            if full_grad is None:
                continue

            group = next(
                group
                for group in self.param_groups
                if any(candidate is param for candidate in group["params"])
            )
            state = self.state[param]
            free_grad = full_grad[:-1]
            if group["maximize"]:
                free_grad = -free_grad
            if "tied_gain_step" not in state:
                state["tied_gain_step"] = 0
                state["tied_gain_exp_avg"] = torch.zeros_like(
                    free_grad
                )
                state["tied_gain_exp_avg_sq"] = torch.zeros_like(
                    free_grad
                )

            state["tied_gain_step"] += 1
            step = state["tied_gain_step"]
            beta1, beta2 = group["betas"]
            exp_avg = state["tied_gain_exp_avg"]
            exp_avg_sq = state["tied_gain_exp_avg_sq"]
            exp_avg.lerp_(free_grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                free_grad, free_grad, value=1.0 - beta2
            )

            lr = group["lr"]
            param.mul_(1.0 - lr * group["weight_decay"])
            step_size = lr / (1.0 - beta1 ** step)
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(1.0 - beta2 ** step)
            ).add_(group["eps"])
            update = exp_avg / denom
            single_count = param.numel() - 1
            param[:single_count].add_(
                update[:single_count], alpha=-step_size
            )
            param[-1].add_(
                update[single_count:].mean(), alpha=-step_size
            )

        for param, grad in zip(self.quotient_params, saved_grads):
>>>>>>> REPLACE

<<<<<<< SEARCH
    embedding_ids = (
        {
            id(param)
            for spec in factor_optimizer.embedding_specs
            for param in spec
        }
        if factor_optimizer is not None
        else set()
    )
    total_sq = None

    for param in parameters:
        if param.grad is None or id(param) in embedding_ids:
=======
    embedding_ids = (
        {
            id(param)
            for spec in factor_optimizer.embedding_specs
            for param in spec
        }
        if factor_optimizer is not None
        else set()
    )
    tied_gain_ids = (
        {
            id(param)
            for param, _ in factor_optimizer.tied_gain_specs
        }
        if factor_optimizer is not None
        else set()
    )
    total_sq = None

    for param in parameters:
        if (
            param.grad is None
            or id(param) in embedding_ids
            or id(param) in tied_gain_ids
        ):
>>>>>>> REPLACE

<<<<<<< SEARCH
    if factor_optimizer is not None:
        for token_param, pos_param in factor_optimizer.embedding_specs:
            term = factor_optimizer.embedding_grad_sq(
                token_param, pos_param
            )
            if term is not None:
                total_sq = (
                    term if total_sq is None else total_sq + term
                )

    for (
=======
    if factor_optimizer is not None:
        for token_param, pos_param in factor_optimizer.embedding_specs:
            term = factor_optimizer.embedding_grad_sq(
                token_param, pos_param
            )
            if term is not None:
                total_sq = (
                    term if total_sq is None else total_sq + term
                )
        for _, owner in factor_optimizer.tied_gain_specs:
            full_grad = owner._ln_f_full_grad
            if full_grad is not None:
                term = full_grad[:-1].square().sum()
                total_sq = (
                    term if total_sq is None else total_sq + term
                )

    for (
>>>>>>> REPLACE

<<<<<<< SEARCH
    clip_coef = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for param in parameters:
        if param.grad is not None:
            param.grad.mul_(clip_coef)


def save_json(path: Path, obj: Dict) -> None:
=======
    clip_coef = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for param in parameters:
        if param.grad is not None:
            param.grad.mul_(clip_coef)
    if factor_optimizer is not None:
        for _, owner in factor_optimizer.tied_gain_specs:
            if owner._ln_f_full_grad is not None:
                owner._ln_f_full_grad.mul_(clip_coef)


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    embedding_specs = [
        (model.token_emb.weight, model.pos_emb.weight)
    ]
    optimizer = QuotientAdamW(
        model.parameters(),
        quotient_params,
        value_bias_specs,
        factor_params,
        embedding_specs=embedding_specs,
        lr=train_cfg.lr,
=======
    embedding_specs = [
        (model.token_emb.weight, model.pos_emb.weight)
    ]
    tied_gain_specs = [
        (model.ln_f_weight, model)
    ]
    optimizer = QuotientAdamW(
        model.parameters(),
        quotient_params,
        value_bias_specs,
        factor_params,
        embedding_specs=embedding_specs,
        tied_gain_specs=tied_gain_specs,
        lr=train_cfg.lr,
>>>>>>> REPLACE