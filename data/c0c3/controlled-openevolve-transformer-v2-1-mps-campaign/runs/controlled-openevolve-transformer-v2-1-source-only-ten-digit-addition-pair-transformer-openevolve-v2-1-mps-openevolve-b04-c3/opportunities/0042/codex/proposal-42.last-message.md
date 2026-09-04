MECHANISM: Two-coordinate cross-head query-bias sharing with lightweight vectorized sampling

HYPOTHESIS: Sharing two corresponding query-bias coordinates will achieve at least 99% accuracy with 1,395 parameters after 75,000 updates, while batch size 128 and exact first-operand-prefiltered rejection will finish within the verification limit.

INTENDED_EDIT: Encode eight query-bias coordinates with six learned scalars, eliminate batch-wide Python holdout checks, reduce the batch size to 128, and validate only at the final positive step.

EVIDENCE: One-coordinate sharing achieved 99.97% with 1,396 parameters; two-coordinate trials produced no negative accuracy evidence at 60,000–75,000 steps but timed out even at batch size 256, whereas shortening to 45,000 steps failed. This motivates preserving 75,000 updates while halving the lightest attempted batch and removing its per-example sampler loop.

<<<<<<< SEARCH
        # Key bias is softmax-invariant, while value bias is absorbed exactly
        # by the affine output projection. Constructing the original Linear
        # first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Independent Q/K rotations let two
        # corresponding query coordinates share values across the heads.
        # Constructing the original Linear first preserves its RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias,
                self.qkv.bias.new_zeros(2 * d_model),
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
    def __init__(
        self,
        batch_size: int,
        seed: int,
        reserved_hashes: set,
        reserved_first: torch.Tensor,
    ):
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.reserved_hashes = reserved_hashes
        self.reserved_first = reserved_first

    def sample_pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)
        b = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)

        # Only examples whose first operand occurs in the holdout can collide.
        slots = torch.searchsorted(self.reserved_first, a)
        in_range = slots < self.reserved_first.numel()
        lookup = slots.clamp(max=self.reserved_first.numel() - 1)
        candidates = torch.nonzero(
            in_range & (self.reserved_first[lookup] == a), as_tuple=False
        ).flatten()
        for i in candidates.tolist():
            ai = int(a[i].item())
            bi = int(b[i].item())
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi
        return a, b
>>>>>>> REPLACE

<<<<<<< SEARCH
    val_a, val_b = splits["val_a"], splits["val_b"]

    model = TinyDecoderLM(model_cfg).to(device)
=======
    val_a, val_b = splits["val_a"], splits["val_b"]
    reserved_first = torch.cat((splits["val_a"], splits["test_a"])).unique(sorted=True)

    model = TinyDecoderLM(model_cfg).to(device)
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
        if (
            (step > 0 and step % train_cfg.eval_interval == 0)
            or (step == train_cfg.train_steps - 1)
        ):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--batch-size", type=int, default=512)
=======
    p.add_argument("--batch-size", type=int, default=128)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=75000)
>>>>>>> REPLACE