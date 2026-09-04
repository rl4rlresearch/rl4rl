MECHANISM: Per-head query-bias rotational gauge fixing with allocation-light holdout filtering

HYPOTHESIS: Fixing one query-bias axis independently in each head will yield a 1,395-parameter model with at least 99% accuracy after 45,000 updates, while exact first-operand prefiltering and final-only validation will finish within the verification limit.

INTENDED_EDIT: Remove one learned query-bias coordinate per head without cross-head coupling, replace batch-wide Python holdout checks with an exact rare-candidate prefilter, and train for 45,000 steps with validation only at the positive final step.

EVIDENCE: The one-coordinate-sharing design achieved 99.97% with 1,396 parameters and reportedly 99.92% at 45,000 steps. The previous independent per-head 1,395-parameter gauge trial timed out at 75,000 steps rather than failing accuracy, motivating the shorter qualified-duration schedule and sampler optimization.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, while value bias is absorbed exactly
        # by the affine output projection. Constructing the original Linear
        # first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, while value bias is absorbed exactly
        # by the affine output projection. Independent rotations of each
        # head's Q/K coordinates allow one query-bias axis per head to be
        # fixed without coupling the heads. Constructing the original Linear
        # first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - n_head))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
=======
        learned_query_bias = self.qkv.bias.view(
            self.n_head, self.head_dim - 1
        )
        query_bias = torch.cat(
            (
                learned_query_bias,
                self.qkv.bias.new_zeros(self.n_head, 1),
            ),
            dim=-1,
        ).reshape(-1)
        qkv_bias = torch.cat(
            (
                query_bias,
                self.qkv.bias.new_zeros(2 * d_model),
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
=======
class TrainBatchSampler:
    def __init__(
        self,
        batch_size: int,
        seed: int,
        reserved_hashes: set,
        reserved_first: set,
    ):
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.reserved_hashes = reserved_hashes
        self.reserved_first = torch.tensor(
            sorted(reserved_first), dtype=torch.int64
        )

    def sample_pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)
        b = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)

        # A reserved pair must first match a reserved first operand. Tensor
        # prefiltering leaves exact hash rejection only for those rare rows.
        candidate_pos = torch.searchsorted(self.reserved_first, a)
        clamped_pos = candidate_pos.clamp_max(self.reserved_first.numel() - 1)
        candidate_mask = (candidate_pos < self.reserved_first.numel()) & (
            self.reserved_first[clamped_pos] == a
        )
        for i in torch.nonzero(candidate_mask, as_tuple=False).flatten().tolist():
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
        if ((step + 1) % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=75000)
=======
    p.add_argument("--train-steps", type=int, default=45000)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=45000)
>>>>>>> REPLACE