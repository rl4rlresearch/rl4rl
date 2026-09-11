MECHANISM: Row-wise positional-embedding quotient under pre-LayerNorm invariance

HYPOTHESIS: Removing each positional embedding row’s feature-uniform degree while preserving its full-coordinate clipping and AdamW difference dynamics will reduce the model to `1638 - INPUT_LEN` parameters and retain at least 99% accuracy.

INTENDED_EDIT: Store seven relative coordinates per positional embedding row, reconstruct eight-coordinate embeddings during forward passes, and apply row-wise quotient-aware gradient clipping and AdamW updates.

EVIDENCE: The current 1638-parameter design achieved 99.96% accuracy, while the 1635-parameter MLP quotient design achieved 99.85%, demonstrating that trajectory-preserving removal of feature-uniform coordinates canceled by downstream LayerNorm can retain accuracy.

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # A feature-uniform offset in any positional row passes unchanged
        # through residual connections and is canceled by every downstream
        # LayerNorm. Store only the row's relative coordinates.
        full_pos_weight = self.pos_emb.weight.detach()
        self.pos_emb.weight = nn.Parameter(
            full_pos_weight[:, :-1] - full_pos_weight[:, -1:]
        )

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
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
        x = self.drop(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
def save_json(path: Path, obj: Dict) -> None:
=======
class RowwiseQuotientAdamW(torch.optim.AdamW):
    """AdamW preserving omitted uniform coordinates row by row."""

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
                full_shape = list(param.shape)
                full_shape[-1] += 1
                state["quotient_step"] = 0
                state["quotient_exp_avg"] = param.new_zeros(full_shape)
                state["quotient_exp_avg_sq"] = param.new_zeros(full_shape)

            full_grad = torch.cat(
                (grad, -grad.sum(dim=-1, keepdim=True)),
                dim=-1,
            )
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
                full_update[..., :-1] - full_update[..., -1:],
                alpha=-step_size,
            )

        return loss


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


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    quotient_params = [model.pos_emb.weight]
    optimizer = RowwiseQuotientAdamW(
        model.parameters(),
        quotient_params,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        if train_cfg.grad_clip > 0:
            clip_rowwise_quotient_grad_norm_(
                model.parameters(), quotient_params, train_cfg.grad_clip
            )
        optimizer.step()
>>>>>>> REPLACE