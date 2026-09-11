MECHANISM: Zero-copy normalized-input chart with gauge-fixed query bias

HYPOTHESIS: A per-head scalar query bias will reduce the model from 1,384 to 1,378 parameters while retaining at least 99% accuracy at 52,000 steps; eliminating basis multiplications, per-example hash checks, and intermediate validation will allow verification to finish.

INTENDED_EDIT: Represent mean-zero LayerNorm outputs by their first seven coordinates, constrain query bias to one scalar per head, accelerate exact holdout exclusion, and validate only at the final positive step.

EVIDENCE: The prior 1,378-parameter query-bias model timed out without contrary accuracy evidence, while the 1,384-parameter model reached 99.99%; repeated timeouts motivate preserving the proven training budget while removing avoidable runtime overhead.

<<<<<<< SEARCH
class MeanZeroInputLinear(nn.Module):
    """Linear map restricted to the mean-zero input subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")

        basis = torch.zeros(in_features, in_features - 1)
        for j in range(in_features - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
        self.linear = nn.Linear(in_features - 1, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x @ self.basis)
=======
class MeanZeroInputLinear(nn.Module):
    """Linear map using a zero-copy chart of the mean-zero input subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")
        self.linear = nn.Linear(in_features - 1, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x[..., :-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = MeanZeroInputLinear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = MeanZeroInputLinear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(n_head))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        q = q + self.q_bias.view(1, self.n_head, 1, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    build_holdout_splits,
    encode_batch,
    pair_hash,
)
=======
    build_holdout_splits,
    encode_batch,
)
>>>>>>> REPLACE

<<<<<<< SEARCH
class TrainBatchSampler:
    def __init__(self, batch_size: int, seed: int, reserved_hashes: set):
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.reserved_hashes = reserved_hashes

    def sample_pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)
        b = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)

        # Strictly avoid holdout leakage.
        for i in range(self.batch_size):
            ai = int(a[i].item())
            bi = int(b[i].item())
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi
        return a, b
=======
class TrainBatchSampler:
    def __init__(self, batch_size: int, seed: int, reserved_by_a: Dict[int, set]):
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.reserved_by_a = reserved_by_a
        self.reserved_a = set(reserved_by_a)

    def sample_pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)
        b = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)

        # The first-operand filter makes the overwhelmingly common path a
        # bulk tensor-to-list conversion and a C-level set intersection.
        a_values = a.tolist()
        candidate_a = self.reserved_a.intersection(a_values)
        if not candidate_a:
            return a, b

        b_values = b.tolist()
        for i, (ai, bi) in enumerate(zip(a_values, b_values)):
            blocked_b = self.reserved_by_a.get(ai)
            while blocked_b is not None and bi in blocked_b:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                blocked_b = self.reserved_by_a.get(ai)
            a[i] = ai
            b[i] = bi
        return a, b
>>>>>>> REPLACE

<<<<<<< SEARCH
    reserved_hashes = set()
    for ai, bi in zip(splits["val_a"].tolist(), splits["val_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))
    for ai, bi in zip(splits["test_a"].tolist(), splits["test_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))

    val_a, val_b = splits["val_a"], splits["val_b"]

    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    reserved_by_a: Dict[int, set] = {}
    for ai, bi in zip(splits["val_a"].tolist(), splits["val_b"].tolist()):
        reserved_by_a.setdefault(int(ai), set()).add(int(bi))
    for ai, bi in zip(splits["test_a"].tolist(), splits["test_b"].tolist()):
        reserved_by_a.setdefault(int(ai), set()).add(int(bi))

    val_a, val_b = splits["val_a"], splits["val_b"]

    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_by_a)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE