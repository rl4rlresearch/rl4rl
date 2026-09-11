MECHANISM: Max-pivot normalized value/output scale gauge

HYPOTHESIS: Fixing only the attention value matrix’s globally redundant norm, while preserving its exact initialized direction and keeping an independent learned positional gain, will achieve at least 99% accuracy with 1,395 parameters after 75,000 updates.

INTENDED_EDIT: Adopt the qualified sinusoidal/RMSNorm/shared-query architecture, encode the value matrix with 63 projective coordinates plus a fixed initialization-derived norm, exempt those coordinates from coordinate-dependent weight decay, accelerate exact holdout filtering, and validate only at the final positive step.

EVIDENCE: The 1,396-parameter shared-query design reached 99.97%. The prior single-weight value anchor scored 0% after 45,000 steps; this patch instead preserves the complete initialized value matrix exactly, fixes the smooth global V/O scale redundancy, leaves positional gain independent, and restores the qualified 75,000-step schedule.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. One query coordinate is shared across the
        # first two heads.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def compress_value_gauge(self) -> None:
        """Replace the redundant value-matrix norm with projective coordinates."""
        qk_weight = self.qkv.weight[: 2 * self.d_model].detach().clone()
        value = self.qkv.weight[2 * self.d_model :].detach().reshape(-1).clone()
        query_bias = self.qkv.bias.detach().clone()

        pivot = int(value.abs().argmax().item())
        keep = torch.cat(
            (
                torch.arange(pivot, device=value.device),
                torch.arange(pivot + 1, value.numel(), device=value.device),
            )
        )
        coords = value[keep] / value[pivot]
        packed_order = torch.cat((keep, keep.new_tensor([pivot])))
        unpack = torch.argsort(packed_order)
        pivot_sign = value[pivot].sign()

        del self.qkv
        self.qk_weight = nn.Parameter(qk_weight)
        self.value_coords = nn.Parameter(coords)
        self.position_scale = nn.Parameter(value.new_ones(()))
        self.query_bias = nn.Parameter(query_bias)
        self.register_buffer("value_unpack", unpack)
        self.register_buffer("value_norm", value.norm())
        self.register_buffer("value_sign", pivot_sign)

    def value_weight(self) -> torch.Tensor:
        packed = torch.cat(
            (
                self.value_coords,
                self.value_coords.new_ones(1),
            )
        )
        direction = self.value_sign * packed[self.value_unpack]
        direction = direction * (self.value_norm / direction.norm().clamp_min(1e-8))
        return direction.view(self.d_model, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        query_bias = torch.cat(
            (
                self.query_bias[: self.head_dim],
                self.query_bias[:1],
                self.query_bias[self.head_dim :],
            )
        )
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.new_zeros(2 * d_model),
            )
        )
        qkv_weight = torch.cat((self.qk_weight, self.value_weight()), dim=0)
        qkv = F.linear(x, qkv_weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
class RMSNorm(nn.Module):
    """Parameter-free normalization that preserves the residual mean channel."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        positions = torch.arange(cfg.max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / cfg.d_model)
        )
        pos_encoding = torch.zeros(cfg.max_seq_len, cfg.d_model)
        pos_encoding[:, 0::2] = torch.sin(positions * frequencies)
        pos_encoding[:, 1::2] = torch.cos(
            positions * frequencies[: pos_encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("pos_encoding", 0.02 * pos_encoding, persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = RMSNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.compress_value_gauge()

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
=======
        position_scale = self.blocks[0].attn.position_scale
        x = self.token_emb(idx) + position_scale * self.pos_encoding[:seqlen].unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
class TrainBatchSampler:
    def __init__(self, batch_size: int, seed: int, reserved_hashes: set):
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.reserved_hashes = reserved_hashes
=======
class TrainBatchSampler:
    def __init__(self, batch_size: int, seed: int, reserved_hashes: set, reserved_first: set):
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.reserved_hashes = reserved_hashes
        self.reserved_first = torch.tensor(sorted(reserved_first), dtype=torch.int64)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Strictly avoid holdout leakage.
        for i in range(self.batch_size):
            ai = int(a[i].item())
            bi = int(b[i].item())
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi
=======
        # A pair can collide only when its first operand occurs in the holdout.
        positions = torch.searchsorted(self.reserved_first, a)
        valid = positions < self.reserved_first.numel()
        candidates = torch.zeros_like(valid)
        candidates[valid] = self.reserved_first[positions[valid]] == a[valid]

        for i in torch.nonzero(candidates, as_tuple=False).flatten().tolist():
            ai = int(a[i].item())
            bi = int(b[i].item())
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi
>>>>>>> REPLACE

<<<<<<< SEARCH
    reserved_hashes = set()
    for ai, bi in zip(splits["val_a"].tolist(), splits["val_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))
    for ai, bi in zip(splits["test_a"].tolist(), splits["test_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))
=======
    reserved_hashes = set()
    reserved_first = set()
    for ai, bi in zip(splits["val_a"].tolist(), splits["val_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))
        reserved_first.add(int(ai))
    for ai, bi in zip(splits["test_a"].tolist(), splits["test_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))
        reserved_first.add(int(ai))
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
=======
    gauge_coords = [block.attn.value_coords for block in model.blocks]
    gauge_ids = {id(parameter) for parameter in gauge_coords}
    regular_parameters = [parameter for parameter in model.parameters() if id(parameter) not in gauge_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": regular_parameters},
            {"params": gauge_coords, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    sampler = TrainBatchSampler(
        train_cfg.batch_size,
        train_cfg.seed + 1337,
        reserved_hashes,
        reserved_first,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if (step > 0 and step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=5000)
=======
    p.add_argument("--train-steps", type=int, default=75000)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=75000)
>>>>>>> REPLACE