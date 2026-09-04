MECHANISM: Multi-query causal attention with shared key/value representation

HYPOTHESIS: Two learned query heads can independently address operand positions through one shared learned key/value space, achieving at least 99% accuracy after 75,000 updates with 1,332 parameters.

INTENDED_EDIT: Replace dense per-head Q/K/V projections with two query heads sharing a four-dimensional key/value projection, retain the qualified one-coordinate query-bias sharing, accelerate exact holdout rejection, and validate only at the final positive step.

EVIDENCE: The 1,396-parameter shared-query design achieved 99.97%, while repeated attempts to remove isolated parameters from embeddings, biases, or matrix gauges failed or timed out. This motivates challenging the load-bearing assumption that both attention heads require independent key and value representations while preserving their separate learned queries and the successful sinusoidal backbone.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, while value bias is absorbed exactly
        # by the affine output projection. Constructing the original Linear
        # first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # The query heads remain independent addressers, but they consult one
        # shared learned key/value representation. This multi-query structure
        # matches the task's common digit alphabet while allowing the heads to
        # retrieve different positions. One query-bias coordinate is shared
        # across the two heads, as in the qualified 1,396-parameter design.
        self.q_proj = nn.Linear(d_model, d_model)
        self.q_proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.kv_proj = nn.Linear(d_model, 2 * self.head_dim, bias=False)
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv_bias = torch.cat(
            (
                self.qkv.bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
=======
        bsz, seqlen, d_model = x.shape
        query_bias = torch.cat(
            (
                self.q_proj.bias[: self.head_dim],
                self.q_proj.bias[:1],
                self.q_proj.bias[self.head_dim :],
            )
        )
        q = F.linear(x, self.q_proj.weight, query_bias)
        k, v = self.kv_proj(x).chunk(2, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, 1, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, 1, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
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
    def __init__(self, batch_size: int, seed: int, reserved_hashes: set, reserved_first: set):
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.reserved_hashes = reserved_hashes
        self.reserved_first = torch.tensor(sorted(reserved_first), dtype=torch.int64)

    def sample_pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)
        b = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)

        # A holdout collision is possible only when the first operand occurs in
        # the holdout. Locate those rare candidates in compiled tensor code,
        # then retain the original exact hash check and rejection semantics.
        candidate_indices = []
        if self.reserved_first.numel() > 0:
            locations = torch.searchsorted(self.reserved_first, a)
            in_range = locations < self.reserved_first.numel()
            safe_locations = locations.clamp_max(self.reserved_first.numel() - 1)
            candidate_indices = torch.nonzero(
                in_range & (self.reserved_first[safe_locations] == a),
                as_tuple=False,
            ).flatten().tolist()

        for i in candidate_indices:
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
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE