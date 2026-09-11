MECHANISM: Joint token-position common-offset quotient

HYPOTHESIS: Quotienting the seven remaining feature coordinates shared between every token row and every positional row will reduce the verified 1514-parameter model to 1507 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Fix the final token-embedding row to zero, transfer its feature offset into every positional row, and preserve full-coordinate embedding AdamW and clipping dynamics with a coupled quotient update.

EVIDENCE: The current 1514-parameter implementation achieved 99.97% after trajectory-preserving input/output quotients. A common token-row shift can be canceled by the opposite positional shift while changing output logits only uniformly across the vocabulary, so this removes seven parameters without restricting the learned function class.

<<<<<<< SEARCH
        # A global feature-uniform shift of every tied token row is canceled
        # at the input and becomes a vocabulary-uniform output-logit shift.
        full_token_weight = self.token_emb.weight.detach().flatten()
        self.token_emb.weight = nn.Parameter(
            full_token_weight[:-1] - full_token_weight[-1]
        )
        self.lm_head.weight = self.token_emb.weight

        # Uniform feature offsets in positional rows are canceled by all
        # downstream LayerNorms. Store only relative coordinates per row.
        full_pos_weight = self.pos_emb.weight.detach()
        self.pos_emb.weight = nn.Parameter(
            full_pos_weight[:, :-1] - full_pos_weight[:, -1:]
        )
=======
        # Shifting every token row by the same feature vector and every
        # positional row by its negative leaves all inputs unchanged and
        # changes output logits only by a vocabulary-uniform scalar. Fix the
        # final token row to zero and transfer its offset into positions.
        full_token_weight = self.token_emb.weight.detach()
        token_offset = full_token_weight[-1:]
        self.token_emb.weight = nn.Parameter(
            full_token_weight[:-1] - token_offset
        )
        self.lm_head.weight = self.token_emb.weight

        # Positional rows also retain only coordinates relative to their
        # final feature, which downstream LayerNorms cancel.
        full_pos_weight = self.pos_emb.weight.detach() + token_offset
        self.pos_emb.weight = nn.Parameter(
            full_pos_weight[:, :-1] - full_pos_weight[:, -1:]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_relative = torch.cat(
            (
                self.token_emb.weight,
                self.token_emb.weight.new_zeros(1),
            )
        )
        token_weight = (
            token_relative + self.token_emb.weight.mean()
        ).view(self.cfg.vocab_size, self.cfg.d_model)
=======
        token_weight = torch.cat(
            (
                self.token_emb.weight,
                self.token_emb.weight.new_zeros(
                    (1, self.cfg.d_model)
                ),
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias_specs=(),
        factor_params=(),
        **kwargs,
    ):
        self.quotient_params = list(quotient_params)
        self.value_bias_specs = list(value_bias_specs)
        self.factor_params = list(factor_params)
        super().__init__(params, **kwargs)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    @torch.no_grad()
    def step(self, closure=None):
=======
    @staticmethod
    def _embedding_full_grads(token_grad, pos_grad):
        full_pos_grad = torch.cat(
            (
                pos_grad,
                -pos_grad.sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        final_token_grad = (
            full_pos_grad.sum(dim=0) - token_grad.sum(dim=0)
        )
        full_token_grad = torch.cat(
            (token_grad, final_token_grad.unsqueeze(0)),
            dim=0,
        )
        return full_token_grad, full_pos_grad

    @torch.no_grad()
    def embedding_grad_sq(self, token_param, pos_param):
        if token_param.grad is None or pos_param.grad is None:
            return None
        full_token_grad, full_pos_grad = self._embedding_full_grads(
            token_param.grad.detach(),
            pos_param.grad.detach(),
        )
        return (
            full_token_grad.square().sum()
            + full_pos_grad.square().sum()
        )

    @torch.no_grad()
    def step(self, closure=None):
>>>>>>> REPLACE

<<<<<<< SEARCH
        saved_grads = [param.grad for param in self.quotient_params]
        saved_factor_grads = [
            param.grad for param in self.factor_params
        ]
        for param in self.quotient_params:
            param.grad = None
        for param in self.factor_params:
            param.grad = None
=======
        saved_grads = [param.grad for param in self.quotient_params]
        saved_factor_grads = [
            param.grad for param in self.factor_params
        ]
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            param.add_(
                full_update[..., :-1] - full_update[..., -1:],
                alpha=-step_size,
            )

        # Reproduce AdamW on each downstream weight and its omitted
=======
            param.add_(
                full_update[..., :-1] - full_update[..., -1:],
                alpha=-step_size,
            )

        # Reproduce independent full-coordinate AdamW updates for token and
        # positional embeddings, then return to their coupled offset gauge.
        for (
            token_param,
            pos_param,
        ), (
            token_grad,
            pos_grad,
        ) in zip(self.embedding_specs, saved_embedding_grads):
            token_param.grad = token_grad
            pos_param.grad = pos_grad
            if token_grad is None or pos_grad is None:
                continue

            group = next(
                group
                for group in self.param_groups
                if any(
                    candidate is token_param
                    for candidate in group["params"]
                )
            )
            state = self.state[token_param]
            full_token_grad, full_pos_grad = (
                self._embedding_full_grads(
                    token_grad.detach(),
                    pos_grad.detach(),
                )
            )
            if group["maximize"]:
                full_token_grad = -full_token_grad
                full_pos_grad = -full_pos_grad

            if "embedding_step" not in state:
                state["embedding_step"] = 0
                state["token_exp_avg"] = torch.zeros_like(
                    full_token_grad
                )
                state["token_exp_avg_sq"] = torch.zeros_like(
                    full_token_grad
                )
                state["pos_exp_avg"] = torch.zeros_like(
                    full_pos_grad
                )
                state["pos_exp_avg_sq"] = torch.zeros_like(
                    full_pos_grad
                )

            state["embedding_step"] += 1
            step = state["embedding_step"]
            beta1, beta2 = group["betas"]
            token_exp_avg = state["token_exp_avg"]
            token_exp_avg_sq = state["token_exp_avg_sq"]
            pos_exp_avg = state["pos_exp_avg"]
            pos_exp_avg_sq = state["pos_exp_avg_sq"]

            token_exp_avg.lerp_(full_token_grad, 1.0 - beta1)
            token_exp_avg_sq.mul_(beta2).addcmul_(
                full_token_grad,
                full_token_grad,
                value=1.0 - beta2,
            )
            pos_exp_avg.lerp_(full_pos_grad, 1.0 - beta1)
            pos_exp_avg_sq.mul_(beta2).addcmul_(
                full_pos_grad,
                full_pos_grad,
                value=1.0 - beta2,
            )

            lr = group["lr"]
            decay = 1.0 - lr * group["weight_decay"]
            token_param.mul_(decay)
            pos_param.mul_(decay)
            step_size = lr / (1.0 - beta1 ** step)
            bias_correction2 = math.sqrt(1.0 - beta2 ** step)
            token_update = token_exp_avg / (
                token_exp_avg_sq.sqrt().div_(
                    bias_correction2
                ).add_(group["eps"])
            )
            pos_update = pos_exp_avg / (
                pos_exp_avg_sq.sqrt().div_(
                    bias_correction2
                ).add_(group["eps"])
            )

            token_param.add_(
                token_update[:-1] - token_update[-1:],
                alpha=-step_size,
            )
            shifted_pos_update = (
                pos_update + token_update[-1].unsqueeze(0)
            )
            pos_param.add_(
                shifted_pos_update[:, :-1]
                - shifted_pos_update[:, -1:],
                alpha=-step_size,
            )

        # Reproduce AdamW on each downstream weight and its omitted
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    quotient_ids = {id(param) for param in quotient_params}
    factor_ids = (
        {id(param) for param in factor_optimizer.factor_params}
        if factor_optimizer is not None
        else set()
    )
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
            continue
        grad = param.grad.detach()
        if id(param) in factor_ids:
            term = factor_optimizer.factor_grad_sq(param)
        else:
            term = grad.square().sum()
            if id(param) in quotient_ids:
                term = term + grad.sum(dim=-1).square().sum()
        total_sq = term if total_sq is None else total_sq + term

    if factor_optimizer is not None:
        for token_param, pos_param in factor_optimizer.embedding_specs:
            term = factor_optimizer.embedding_grad_sq(
                token_param, pos_param
            )
            if term is not None:
                total_sq = (
                    term if total_sq is None else total_sq + term
                )

    for qkv_bias, proj_weight, proj_bias in value_bias_specs:
>>>>>>> REPLACE

<<<<<<< SEARCH
    quotient_params = [model.token_emb.weight, model.pos_emb.weight] + [
        block.attn.proj.bias for block in model.blocks
    ] + [
=======
    quotient_params = [
        block.attn.proj.bias for block in model.blocks
    ] + [
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    factor_params = [
        block.attn.qkv.weight for block in model.blocks
    ] + [
        block.mlp.fc1.weight for block in model.blocks
    ]
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
>>>>>>> REPLACE