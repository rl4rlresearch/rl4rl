MECHANISM: Final-LayerNorm bias-coordinate ablation atop the qualified joint quotients

HYPOTHESIS: Reproducing the verified 1506-parameter design and fixing the last final-LayerNorm bias coordinate to zero will yield 1505 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Apply the qualified weight, residual-output, and token-position quotients with their trajectory-aware optimizer; retain the final-LayerNorm scale gauge; and store seven rather than eight final-LayerNorm bias coordinates.

EVIDENCE: Reference Design 3 achieved 99.96% accuracy at 1506 parameters. Removing one final-affine bias coordinate is the smallest remaining capacity restriction and directly tests whether that margin survives a one-parameter reduction.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, and the complete value bias can be
        # absorbed by the downstream projection bias. Store only query bias.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, and every value-bias coordinate can
        # be absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
        # Downstream LayerNorms cancel its feature-uniform coordinate.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                value_bias,
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
=======
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                value_bias,
            )
        )
        qkv_weight_relative = torch.cat(
            (
                self.qkv.weight,
                self.qkv.weight.new_zeros(
                    (self.qkv.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        qkv_weight = (
            qkv_weight_relative
            + self.qkv.weight.mean(dim=-1, keepdim=True)
        )
        qkv = F.linear(x, qkv_weight, qkv_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        weight_relative = torch.cat(
            (
                self.proj.weight,
                self.proj.weight.new_zeros(
                    (self.proj.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        proj_weight = (
            weight_relative
            + self.proj.weight.mean(dim=-1, keepdim=True)
        ).transpose(0, 1)
        relative_bias = torch.cat(
            (self.proj.bias, self.proj.bias.new_zeros(1))
        )
        proj_bias = relative_bias + self.proj.bias.mean()
        y = F.linear(y, proj_weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # The final LayerNorm cancels the feature-uniform component.
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        relative_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        fc2_bias = relative_bias + self.fc2.bias.mean()
        return self.drop(F.linear(hidden, self.fc2.weight, fc2_bias))
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # The final LayerNorm cancels the feature-uniform component.
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight_relative = torch.cat(
            (
                self.fc1.weight,
                self.fc1.weight.new_zeros(
                    (self.fc1.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        fc1_weight = (
            fc1_weight_relative
            + self.fc1.weight.mean(dim=-1, keepdim=True)
        )
        hidden = F.gelu(F.linear(x, fc1_weight, self.fc1.bias))
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
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
=======
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        # A common positive scale changes only the global logit temperature.
        # The final bias additionally fixes its last coordinate to zero.
        self.ln_f = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
        self.ln_f_weight = nn.Parameter(torch.ones(cfg.d_model - 1))
        self.ln_f_bias = nn.Parameter(torch.zeros(cfg.d_model - 1))

        # Weight tying with input embeddings.
>>>>>>> REPLACE

<<<<<<< SEARCH
        # A globally uniform tied-embedding coordinate is canceled at the
        # input and produces only a vocabulary-uniform output-logit shift.
        full_token_weight = self.token_emb.weight.detach().flatten()
        self.token_emb.weight = nn.Parameter(
            full_token_weight[:-1] - full_token_weight[-1]
        )
        self.lm_head.weight = self.token_emb.weight

        # A feature-uniform offset in any positional row passes unchanged
        # through residual connections and is canceled by every downstream
        # LayerNorm. Store only the row's relative coordinates.
        full_pos_weight = self.pos_emb.weight.detach()
        self.pos_emb.weight = nn.Parameter(
            full_pos_weight[:, :-1] - full_pos_weight[:, -1:]
        )
=======
        # Retain LayerNorm-null input and residual-output coordinates only
        # through relative representatives.
        for block in self.blocks:
            full_qkv_weight = block.attn.qkv.weight.detach()
            block.attn.qkv.weight = nn.Parameter(
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            full_proj_weight = (
                block.attn.proj.weight.detach().transpose(0, 1)
            )
            block.attn.proj.weight = nn.Parameter(
                full_proj_weight[:, :-1] - full_proj_weight[:, -1:]
            )
            full_fc1_weight = block.mlp.fc1.weight.detach()
            block.mlp.fc1.weight = nn.Parameter(
                full_fc1_weight[:, :-1] - full_fc1_weight[:, -1:]
            )
            full_fc2_weight = (
                block.mlp.fc2.weight.detach().transpose(0, 1)
            )
            block.mlp.fc2.weight = nn.Parameter(
                full_fc2_weight[:, :-1] - full_fc2_weight[:, -1:]
            )

        # Fix the final token row to zero and transfer its common offset into
        # every positional row.
        full_token_weight = self.token_emb.weight.detach()
        token_offset = full_token_weight[-1:]
        self.token_emb.weight = nn.Parameter(
            full_token_weight[:-1] - token_offset
        )
        self.lm_head.weight = self.token_emb.weight

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
        x = self.ln_f(x)
        logits = F.linear(x, token_weight)
=======
        x = self.ln_f(x)
        ln_f_weight = torch.cat(
            (
                self.ln_f_weight,
                self.ln_f_weight.new_ones(1),
            )
        )
        ln_f_bias = torch.cat(
            (
                self.ln_f_bias,
                self.ln_f_bias.new_zeros(1),
            )
        )
        x = x * ln_f_weight + ln_f_bias
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
class RowwiseQuotientAdamW(torch.optim.AdamW):
    """AdamW preserving omitted uniform and absorbed-bias coordinates."""

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
=======
def reconstruct_input_weight(param):
    relative = torch.cat(
        (
            param,
            param.new_zeros((param.size(0), 1)),
        ),
        dim=-1,
    )
    return relative + param.mean(dim=-1, keepdim=True)


def reconstruct_output_weight(param):
    return reconstruct_input_weight(param).transpose(0, 1)


class QuotientAdamW(torch.optim.AdamW):
    """AdamW preserving omitted uniform and absorbed-bias coordinates."""

    def __init__(
        self,
        params,
        quotient_params,
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
=======
    def _factor_state(self, param):
        state = self.state[param]
        if "factor_weight" not in state:
            state["factor_step"] = 0
            state["factor_weight"] = reconstruct_input_weight(
                param.detach()
            ).clone()
            state["factor_scale"] = param.new_ones(param.size(1) + 1)
            state["factor_weight_exp_avg"] = torch.zeros_like(
                state["factor_weight"]
            )
            state["factor_weight_exp_avg_sq"] = torch.zeros_like(
                state["factor_weight"]
            )
            state["factor_scale_exp_avg"] = param.new_zeros(
                param.size(1) + 1
            )
            state["factor_scale_exp_avg_sq"] = param.new_zeros(
                param.size(1) + 1
            )
        return state
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    @torch.no_grad()
    def factor_grad_sq(self, param):
        state = self._factor_state(param)
        grad = param.grad.detach()
        full_grad = torch.cat(
            (grad, -grad.sum(dim=-1, keepdim=True)),
            dim=-1,
        )
        weight_grad = (
            full_grad * state["factor_scale"].unsqueeze(0)
        )
        scale_grad = (
            full_grad * state["factor_weight"]
        ).sum(dim=0)
        return weight_grad.square().sum() + scale_grad.square().sum()

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
        saved_factor_grads = [
            param.grad for param in self.factor_params
        ]
        for param in self.quotient_params:
            param.grad = None
        for param in self.factor_params:
            param.grad = None

        loss = super().step(closure)
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
                full_proj_weight = reconstruct_output_weight(
                    proj_weight.detach()
                )
                grad = (
                    full_proj_weight
                    * full_proj_grad.unsqueeze(1)
                ).sum(dim=0)
            value_bias_grads.append(grad)

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

        loss = super().step(closure)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Reproduce AdamW on each downstream weight and its omitted
        # LayerNorm scale, then store their sufficient columnwise product.
=======
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
        # LayerNorm scale, then store their sufficient columnwise product.
>>>>>>> REPLACE

<<<<<<< SEARCH
            factor_weight = state["factor_weight"]
            factor_scale = state["factor_scale"]
            weight_grad = grad * factor_scale.unsqueeze(0)
            scale_grad = (grad * factor_weight).sum(dim=0)
            if group["maximize"]:
                weight_grad = -weight_grad
                scale_grad = -scale_grad
=======
            factor_weight = state["factor_weight"]
            factor_scale = state["factor_scale"]
            full_grad = torch.cat(
                (grad, -grad.sum(dim=-1, keepdim=True)),
                dim=-1,
            )
            weight_grad = full_grad * factor_scale.unsqueeze(0)
            scale_grad = (full_grad * factor_weight).sum(dim=0)
            if group["maximize"]:
                weight_grad = -weight_grad
                scale_grad = -scale_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
            param.copy_(
                factor_weight * factor_scale.unsqueeze(0)
            )
=======
            full_product = (
                factor_weight * factor_scale.unsqueeze(0)
            )
            param.copy_(
                full_product[:, :-1] - full_product[:, -1:]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            omitted_value = -step_size * exp_avg / denom
            proj_bias.add_(
                proj_weight[:, -omitted_value.numel():] @ omitted_value
            )
=======
            omitted_value = -step_size * exp_avg / denom
            full_delta = (
                reconstruct_output_weight(proj_weight)
                @ omitted_value
            )
            proj_bias.add_(
                full_delta[:-1] - full_delta[-1]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        total_sq = term if total_sq is None else total_sq + term

    for qkv_bias, proj_weight, proj_bias in value_bias_specs:
        if proj_bias.grad is None:
            continue
        omitted_dims = 2 * proj_weight.size(1) - qkv_bias.numel()
        omitted_grad = (
            proj_weight.detach()[:, -omitted_dims:]
            * proj_bias.grad.detach().unsqueeze(1)
        ).sum(dim=0)
=======
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
        if proj_bias.grad is None:
            continue
        full_proj_grad = torch.cat(
            (
                proj_bias.grad.detach(),
                -proj_bias.grad.detach().sum().view(1),
            )
        )
        full_proj_weight = reconstruct_output_weight(
            proj_weight.detach()
        )
        omitted_grad = (
            full_proj_weight
            * full_proj_grad.unsqueeze(1)
        ).sum(dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    quotient_params = [model.token_emb.weight, model.pos_emb.weight] + [
        block.mlp.fc2.bias for block in model.blocks
    ]
=======
    quotient_params = [
        block.attn.proj.bias for block in model.blocks
    ] + [
        block.attn.proj.weight for block in model.blocks
    ] + [
        block.mlp.fc2.bias for block in model.blocks
    ] + [
        block.mlp.fc2.weight for block in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer = RowwiseQuotientAdamW(
        model.parameters(),
        quotient_params,
        value_bias_specs,
        factor_params,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
=======
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
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
            clip_rowwise_quotient_grad_norm_(
                model.parameters(),
                quotient_params,
                value_bias_specs,
                train_cfg.grad_clip,
                optimizer,
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