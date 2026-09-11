MECHANISM: Quotient-aware value-bias absorption into the attention output bias

HYPOTHESIS: Starting from the qualified 1595-parameter design, removing one attention value-bias coordinate and projecting its full-coordinate AdamW update into `attn.proj.bias` will produce a 1594-parameter model with at least 99% accuracy.

INTENDED_EDIT: Adopt the qualified token, positional, MLP-output, key-bias, and biasless-block-LayerNorm reductions; then store seven value-bias coordinates and preserve the omitted eighth coordinate’s gradient clipping and AdamW effect through the downstream attention projection bias.

EVIDENCE: The 1595-parameter design achieved 99.42% after downstream biases successfully absorbed both block LayerNorm offsets, while trajectory-preserving quotient optimization qualified for positional, token, and MLP-output gauges; an attention value bias is likewise passed unchanged through normalized attention and can be absorbed by the existing projection bias.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Remove six softmax-invariant key-bias degrees while preserving
        # fused-projection construction and a learned shared key-bias value.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 6))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain seven value-bias coordinates; the omitted eighth value
        # coordinate is absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1].expand(6),
                self.qkv.bias[d_model + 1 : 2 * d_model - 6],
                self.qkv.bias[2 * d_model - 6 :],
            )
        )
=======
        query_bias = self.qkv.bias[:d_model]
        value_bias = torch.cat(
            (self.qkv.bias[d_model:], self.qkv.bias.new_zeros(1))
        )
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                value_bias,
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln1.bias = None
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.bias = None
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # A feature-uniform offset in any positional row passes unchanged
        # through residual connections and is canceled by every downstream
        # LayerNorm. Store only the row's relative coordinates.
        full_pos_weight = self.pos_emb.weight.detach()
        self.pos_emb.weight = nn.Parameter(
            full_pos_weight[:, :-1] - full_pos_weight[:, -1:]
        )
=======
        self.apply(self._init_weights)

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
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_relative = torch.cat(
            (
                self.pos_emb.weight,
                self.pos_emb.weight.new_zeros(
                    (self.pos_emb.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        pos_weight = pos_relative + self.pos_emb.weight.mean(
            dim=-1, keepdim=True
        )
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)

        token_relative = torch.cat(
            (
                self.token_emb.weight,
                self.token_emb.weight.new_zeros(1),
            )
        )
        token_weight = (
            token_relative + self.token_emb.weight.mean()
        ).view(self.cfg.vocab_size, self.cfg.d_model)

        pos_relative = torch.cat(
            (
                self.pos_emb.weight,
                self.pos_emb.weight.new_zeros(
                    (self.pos_emb.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        pos_weight = pos_relative + self.pos_emb.weight.mean(
            dim=-1, keepdim=True
        )
        x = F.embedding(idx, token_weight) + F.embedding(pos, pos_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
class RowwiseQuotientAdamW(torch.optim.AdamW):
    """AdamW preserving omitted uniform coordinates row by row."""

    def __init__(self, params, quotient_params, **kwargs):
        self.quotient_params = list(quotient_params)
        super().__init__(params, **kwargs)
=======
class RowwiseQuotientAdamW(torch.optim.AdamW):
    """AdamW preserving omitted uniform and absorbed-bias coordinates."""

    def __init__(
        self, params, quotient_params, value_bias_specs=(), **kwargs
    ):
        self.quotient_params = list(quotient_params)
        self.value_bias_specs = list(value_bias_specs)
        super().__init__(params, **kwargs)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @torch.no_grad()
    def step(self, closure=None):
        saved_grads = [param.grad for param in self.quotient_params]
=======
    @torch.no_grad()
    def step(self, closure=None):
        value_bias_grads = []
        for _, proj_weight, proj_bias in self.value_bias_specs:
            grad = None
            if proj_bias.grad is not None:
                grad = (
                    proj_weight.detach()[:, -1]
                    * proj_bias.grad.detach()
                ).sum()
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

        # Update the omitted value-bias coordinate in full-coordinate AdamW,
        # then immediately return to the zero-coordinate gauge by folding its
        # effect into the already-updated attention projection bias.
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
                state["value_quotient_exp_avg"] = qkv_bias.new_zeros(())
                state["value_quotient_exp_avg_sq"] = qkv_bias.new_zeros(())

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
            proj_bias.add_(proj_weight[:, -1] * omitted_value)

        return loss
>>>>>>> REPLACE

<<<<<<< SEARCH
@torch.no_grad()
def clip_rowwise_quotient_grad_norm_(
    parameters, quotient_params, max_norm: float
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

    if total_sq is None:
        return

    clip_coef = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for param in parameters:
        if param.grad is not None:
            param.grad.mul_(clip_coef)
=======
@torch.no_grad()
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

    for _, proj_weight, proj_bias in value_bias_specs:
        if proj_bias.grad is None:
            continue
        omitted_grad = (
            proj_weight.detach()[:, -1] * proj_bias.grad.detach()
        ).sum()
        term = omitted_grad.square()
        total_sq = term if total_sq is None else total_sq + term

    if total_sq is None:
        return

    clip_coef = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for param in parameters:
        if param.grad is not None:
            param.grad.mul_(clip_coef)
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    quotient_params = [model.pos_emb.weight]
    optimizer = RowwiseQuotientAdamW(
        model.parameters(),
        quotient_params,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
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
    optimizer = RowwiseQuotientAdamW(
        model.parameters(),
        quotient_params,
        value_bias_specs,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
            clip_rowwise_quotient_grad_norm_(
                model.parameters(), quotient_params, train_cfg.grad_clip
            )
=======
            clip_rowwise_quotient_grad_norm_(
                model.parameters(),
                quotient_params,
                value_bias_specs,
                train_cfg.grad_clip,
            )
>>>>>>> REPLACE