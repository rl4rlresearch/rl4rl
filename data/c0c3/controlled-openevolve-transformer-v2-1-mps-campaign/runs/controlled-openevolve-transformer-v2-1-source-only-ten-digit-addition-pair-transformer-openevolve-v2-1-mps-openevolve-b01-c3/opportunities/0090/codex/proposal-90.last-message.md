MECHANISM: Half-width content-addressing with high-resolution positional attention

HYPOTHESIS: A 1,211-parameter transformer will achieve at least 99% accuracy because two-dimensional query/key features should suffice to distinguish the small token vocabulary, while the 99.98%-accurate eleven-bin design preserves head-specific positional routing.

INTENDED_EDIT: Replace four-dimensional query/key vectors with independent two-dimensional vectors while retaining four-dimensional values, and restore the robust eleven-bin far-distance sharing layout.

EVIDENCE: The 1,275-parameter eleven-bin design achieved 99.98%, whereas recent 1,269-parameter failures constrained positional biases, normalization, MLP biases, or residual projections. This tests the previously shared assumption that content addressing needs the full value width while preserving those load-bearing mechanisms.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.proj = PairTiedBiasMeanZeroLinear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 15))
        self.far_rel_bias = nn.Parameter(torch.zeros(14))
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.query_dim = max(1, self.head_dim // 2)
        self.qk = nn.Linear(d_model, 2 * n_head * self.query_dim, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.proj = PairTiedBiasMeanZeroLinear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
        self.far_rel_bias = nn.Parameter(torch.zeros(11))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias = F.pad(
            self.q_bias.expand(self.n_head, 1), (0, self.head_dim - 1)
        ).reshape(d_model)
        q = q + q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
=======
        bsz, seqlen, d_model = x.shape
        q, k = self.qk(x).chunk(2, dim=-1)
        v = self.value(x)
        q_bias = F.pad(
            self.q_bias.expand(self.n_head, 1), (0, self.query_dim - 1)
        ).reshape(self.n_head * self.query_dim)
        q = q + q_bias

        q = q.view(bsz, seqlen, self.n_head, self.query_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.query_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.query_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.rel_bias,
                self.far_rel_bias.expand(self.n_head, 14),
=======
                self.rel_bias,
                self.far_rel_bias.expand(self.n_head, 11),
>>>>>>> REPLACE