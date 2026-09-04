MECHANISM: Two-coordinate cross-head query-bias sharing with sorted holdout prefilter

HYPOTHESIS: Sharing a second corresponding query-bias coordinate will achieve at least 99% accuracy with 1,395 parameters after 75,000 updates, while a vectorized first-operand holdout prefilter will eliminate the sampler’s per-example Python loop and allow training to finish.

INTENDED_EDIT: Encode eight query-bias coordinates with six learned scalars, check only samples whose first operand occurs in the holdout via `searchsorted`, and validate only at positive interval endpoints.

EVIDENCE: One-coordinate sharing achieved 99.97% accuracy with 1,396 parameters. All second-coordinate experiments timed out rather than producing negative accuracy evidence; their remaining sampler still performed batch-wide Python work, motivating an exact tensor prefilter that reduces Python hash checks to the extremely rare first-operand matches.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. One query coordinate is shared across the
        # first two heads; independent Q/K rotations make that equality a
        # one-dimensional gauge choice while keeping every coordinate adaptive.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Two corresponding query coordinates are
        # shared across the first two heads while remaining learned.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = torch.cat(
            (
                self.qkv.bias[: self.head_dim],
                self.qkv.bias[:1],
                self.qkv.bias[self.head_dim :],
            )
        )
=======
        query_bias = torch.cat(
            (
                self.qkv.bias[: self.head_dim],
                self.qkv.bias[:2],
                self.qkv.bias[self.head_dim :],
            )
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
    def __init__(self, batch_size: int, seed: int, reserved_hashes: set, reserved_firsts: set):
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.reserved_hashes = reserved_hashes
        self.reserved_firsts = torch.tensor(sorted(reserved_firsts), dtype=torch.int64)

    def sample_pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)
        b = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)

        # A pair can be reserved only if its first component occurs in the
        # holdout. Find those rare candidates in tensor code, then perform the
        # exact Python hash check only for candidates.
        if self.reserved_firsts.numel() == 0:
            return a, b

        while True:
            positions = torch.searchsorted(self.reserved_firsts, a)
            in_range = positions < self.reserved_firsts.numel()
            safe_positions = positions.clamp_max(self.reserved_firsts.numel() - 1)
            candidates = torch.nonzero(
                in_range & (self.reserved_firsts[safe_positions] == a),
                as_tuple=False,
            ).flatten()

            collisions = [
                int(i)
                for i in candidates.tolist()
                if pair_hash(int(a[i].item()), int(b[i].item())) in self.reserved_hashes
            ]
            if not collisions:
                return a, b

            for i in collisions:
                a[i] = torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64)
                b[i] = torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64)
>>>>>>> REPLACE

<<<<<<< SEARCH
    reserved_hashes = set()
    for ai, bi in zip(splits["val_a"].tolist(), splits["val_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))
    for ai, bi in zip(splits["test_a"].tolist(), splits["test_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))
=======
    reserved_hashes = set()
    reserved_firsts = set()
    for ai, bi in zip(splits["val_a"].tolist(), splits["val_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))
        reserved_firsts.add(int(ai))
    for ai, bi in zip(splits["test_a"].tolist(), splits["test_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))
        reserved_firsts.add(int(ai))
>>>>>>> REPLACE

<<<<<<< SEARCH
    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    sampler = TrainBatchSampler(
        train_cfg.batch_size,
        train_cfg.seed + 1337,
        reserved_hashes,
        reserved_firsts,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if (((step + 1) % train_cfg.eval_interval == 0) or
                (step == train_cfg.train_steps - 1)):
>>>>>>> REPLACE