MECHANISM: Dual residual-output quotient biases with AdamW trajectory preservation

HYPOTHESIS: Removing the feature-uniform degree from both MLP and attention output biases will produce a 1634-parameter model with at least 99% accuracy, because both uniform residual shifts are canceled by downstream LayerNorms and quotient-aware optimization preserves their full-bias AdamW dynamics.

INTENDED_EDIT: Adopt the qualified query-derived key bias, reduce both `fc2.bias` and `attn.proj.bias` to seven relative coordinates, reconstruct their eight-coordinate biases during projection, and train them with quotient-aware clipping and AdamW updates.

EVIDENCE: The 1635-parameter MLP quotient design achieved 99.85%, demonstrating that trajectory-preserving removal of a LayerNorm-invariant bias degree works; the attention output bias has the same exact uniform-shift invariance, motivating the next one-parameter reduction.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Share all softmax-invariant key-bias coordinates as one learned
        # scalar while preserving fused-projection construction.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model + 1))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Reuse the mean learned query bias across every softmax-invariant
        # key-bias coordinate, leaving only query and value bias parameters.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
        # A feature-uniform attention-output bias is canceled by the
        # following LayerNorms, so retain only relative coordinates.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias[d_model : d_model + 1].expand(d_model),
                self.qkv.bias[d_model + 1 :],
            )
        )
=======
        query_bias = self.qkv.bias[:d_model]
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                self.qkv.bias[d_model:],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        relative_bias = torch.cat(
            (self.proj.bias, self.proj.bias.new_zeros(1))
        )
        proj_bias = relative_bias + self.proj.bias.mean()
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
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
        # The final LayerNorm cancels the feature-uniform component of this
        # residual bias, so retain only its relative coordinates.
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
def save_json(path: Path, obj: Dict) -> None:
=======
class QuotientAdamW(torch.optim.AdamW):
    """AdamW preserving omitted uniform-bias coordinates' dynamics."""

    def __init__(self, params, quotient_params, **kwargs):
        self.quotient_params = list(quotient_params)
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        saved_grads = [param.grad for param in self.quotient_params]
        for param in self.quotient_params:
            param.grad = None

        loss = super().step(closure)

        for param, grad in zip(self.quotient_params, saved_grads):
            param.grad = grad
            if grad is None:
                continue

            group = next(
                group
                for group in self.param_groups
                if any(candidate is param for candidate in group["params"])
            )
            state = self.state[param]
            if "quotient_step" not in state:
                state["quotient_step"] = 0
                state["quotient_exp_avg"] = param.new_zeros(param.numel() + 1)
                state["quotient_exp_avg_sq"] = param.new_zeros(param.numel() + 1)

            full_grad = torch.cat((grad, -grad.sum().reshape(1)))
            if group["maximize"]:
                full_grad = -full_grad

            state["quotient_step"] += 1
            step = state["quotient_step"]
            beta1, beta2 = group["betas"]
            exp_avg = state["quotient_exp_avg"]
            exp_avg_sq = state["quotient_exp_avg_sq"]

            exp_avg.lerp_(full_grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                full_grad, full_grad, value=1.0 - beta2
            )

            lr = group["lr"]
            param.mul_(1.0 - lr * group["weight_decay"])
            step_size = lr / (1.0 - beta1 ** step)
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(1.0 - beta2 ** step)
            ).add_(group["eps"])
            full_update = exp_avg / denom
            param.add_(
                full_update[:-1] - full_update[-1],
                alpha=-step_size,
            )

        return loss


@torch.no_grad()
def clip_quotient_grad_norm_(parameters, quotient_params, max_norm: float) -> None:
    parameters = list(parameters)
    quotient_ids = {id(param) for param in quotient_params}
    total_sq = None

    for param in parameters:
        if param.grad is None:
            continue
        term = param.grad.detach().square().sum()
        if id(param) in quotient_ids:
            term = term + param.grad.detach().sum().square()
        total_sq = term if total_sq is None else total_sq + term

    if total_sq is None:
        return

    clip_coef = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for param in parameters:
        if param.grad is not None:
            param.grad.mul_(clip_coef)


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    quotient_params = [
        param
        for block in model.blocks
        for param in (block.attn.proj.bias, block.mlp.fc2.bias)
    ]
    optimizer = QuotientAdamW(
        model.parameters(),
        quotient_params,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        if train_cfg.grad_clip > 0:
            clip_quotient_grad_norm_(
                model.parameters(), quotient_params, train_cfg.grad_clip
            )
        optimizer.step()
>>>>>>> REPLACE