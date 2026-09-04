MECHANISM: Quotient-space positional embeddings with ambient-coordinate AdamW

HYPOTHESIS: Training all mean-zero positional embeddings with projected eight-coordinate AdamW moments will retain at least 99% accuracy with 1,605 parameters, because it removes only LayerNorm-invariant directions while preserving the verified 1,628-parameter model’s initialization and optimizer dynamics.

INTENDED_EDIT: Apply the verified key-bias and `ln2`-bias removals, represent each positional embedding in a seven-dimensional orthonormal mean-zero basis, and update those coordinates using projected full-space AdamW moments.

EVIDENCE: The 1,628-parameter design achieved 99.95%, while naïve mean-zero positional compression reached only 72.91%; this tests whether AdamW’s coordinate-dependent moments—not lost model capacity—caused that exact-invariance reparameterization to fail.

<<<<<<< SEARCH
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
    vocab_size: int


class MeanZeroPositionalEmbedding(nn.Module):
    """Positional embeddings modulo LayerNorm-invariant all-ones shifts."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 1))

        basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for col in range(embedding_dim - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # Draw the same number of values as a full embedding so subsequent
        # module initialization retains the reference model's RNG stream.
        full = torch.empty(
            self.num_embeddings,
            self.embedding_dim,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        nn.init.normal_(full, mean=0.0, std=0.02)
        self.weight.copy_(full @ self.basis)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        full_weight = self.weight @ self.basis.transpose(0, 1)
        return F.embedding(indices, full_weight)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = MeanZeroPositionalEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, MeanZeroPositionalEmbedding):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class TrainBatchSampler:
=======
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class AmbientProjectedAdamW(torch.optim.Optimizer):
    """AdamW in the full embedding coordinates, projected onto the quotient."""

    def __init__(
        self,
        param: torch.nn.Parameter,
        basis: torch.Tensor,
        lr: float,
        weight_decay: float,
    ):
        defaults = {
            "lr": lr,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": weight_decay,
        }
        super().__init__([param], defaults)
        self.basis = basis

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        basis_t = self.basis.transpose(0, 1)
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AmbientProjectedAdamW does not support sparse gradients")

                grad_full = param.grad @ basis_t
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad_full)
                    state["exp_avg_sq"] = torch.zeros_like(grad_full)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad_full, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_full, grad_full, value=1.0 - beta2)

                step = state["step"]
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group["eps"])
                update_full = exp_avg / denom

                param.mul_(1.0 - group["lr"] * group["weight_decay"])
                param.add_(
                    update_full @ self.basis,
                    alpha=-group["lr"] / bias_correction1,
                )
        return loss


class TrainBatchSampler:
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

    pos_weight = model.pos_emb.weight
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p is not pos_weight],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    pos_optimizer = AmbientProjectedAdamW(
        pos_weight,
        model.pos_emb.basis,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    optimizers = (optimizer, pos_optimizer)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
=======
        for opt in optimizers:
            for pg in opt.param_groups:
                pg["lr"] = lr_now
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
=======
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.step()

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        for opt in optimizers:
            opt.step()

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE