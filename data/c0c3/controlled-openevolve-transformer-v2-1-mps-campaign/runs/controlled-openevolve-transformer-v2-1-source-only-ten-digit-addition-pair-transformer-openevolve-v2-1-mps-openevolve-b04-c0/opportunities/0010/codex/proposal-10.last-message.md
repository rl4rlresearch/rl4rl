MECHANISM: Quotient-space positional embeddings with full-coordinate AdamW dynamics

HYPOTHESIS: Removing the unobservable all-ones component from every positional embedding will reduce parameters by `max_seq_len` while retaining at least 99% accuracy, because per-position scalar shifts are erased by every pre-LayerNorm and the final LayerNorm, and the custom optimizer preserves the original eight-coordinate AdamW updates in the seven-dimensional quotient space.

INTENDED_EDIT: Represent each 8-dimensional positional embedding with seven orthonormal zero-mean coordinates, preserve baseline-equivalent initialization, and optimize those coordinates by projecting virtual full-coordinate AdamW updates.

EVIDENCE: The 1628-parameter model reached 99.76%, while deleting or tying functionally absorbable bias pathways caused severe optimization failures; this patch instead removes a strictly unobservable positional direction and explicitly retains the original optimizer dynamics.

<<<<<<< SEARCH
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class GaugePositionEmbedding(nn.Module):
    """Positional embeddings modulo their unobservable all-ones direction."""

    def __init__(self, num_embeddings: int, d_model: int):
        super().__init__()
        if d_model < 2:
            raise ValueError("d_model must be at least 2")

        basis = torch.zeros(d_model, d_model - 1)
        for j in range(d_model - 1):
            denom = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / denom
            basis[j + 1, j] = -(j + 1) / denom
        self.register_buffer("basis", basis, persistent=False)

        # Consume the same random draw as a full nn.Embedding constructor.
        full = torch.empty(num_embeddings, d_model)
        nn.init.normal_(full)
        self.weight = nn.Parameter(full @ basis)
        self.num_embeddings = num_embeddings
        self.d_model = d_model

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        latent = F.embedding(positions, self.weight)
        return latent @ self.basis.transpose(0, 1)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugePositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
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
        if isinstance(module, GaugePositionEmbedding):
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.d_model,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(full @ module.basis)
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


class GaugeAdamW:
    """AdamW on virtual full positional coordinates, stored in quotient form."""

    def __init__(
        self,
        param: torch.nn.Parameter,
        basis: torch.Tensor,
        lr: float,
        weight_decay: float,
    ):
        self.param = param
        self.basis = basis
        self.weight_decay = weight_decay
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.step_count = 0
        full_shape = (param.shape[0], basis.shape[0])
        self.exp_avg = torch.zeros(full_shape, device=param.device, dtype=param.dtype)
        self.exp_avg_sq = torch.zeros_like(self.exp_avg)
        self.param_groups = [{"lr": lr}]

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.param.grad is not None:
            if set_to_none:
                self.param.grad = None
            else:
                self.param.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        if self.param.grad is None:
            return

        grad_full = self.param.grad @ self.basis.transpose(0, 1)
        self.step_count += 1
        self.exp_avg.mul_(self.beta1).add_(grad_full, alpha=1.0 - self.beta1)
        self.exp_avg_sq.mul_(self.beta2).addcmul_(
            grad_full, grad_full, value=1.0 - self.beta2
        )

        bias_correction1 = 1.0 - self.beta1 ** self.step_count
        bias_correction2 = 1.0 - self.beta2 ** self.step_count
        denom = self.exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
        update_full = (self.exp_avg / bias_correction1) / (denom + self.eps)

        lr = self.param_groups[0]["lr"]
        self.param.mul_(1.0 - lr * self.weight_decay)
        self.param.add_(update_full @ self.basis, alpha=-lr)


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
    pos_param = model.pos_emb.weight
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p is not pos_param),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    pos_optimizer = GaugeAdamW(
        pos_param,
        model.pos_emb.basis,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
=======
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        pos_optimizer.param_groups[0]["lr"] = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        pos_optimizer.zero_grad(set_to_none=True)
        loss.backward()
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.step()

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        optimizer.step()
        pos_optimizer.step()

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE