MECHANISM: Head-shared scalar key addressing

HYPOTHESIS: A 1,171-parameter transformer will achieve at least 99% accuracy because the verified 1,179-parameter scalar-query/key model achieved 100%, while sharing only the eight-parameter key projection preserves head-specific queries, values, and positional routing.

INTENDED_EDIT: Replace full-width QKV with one scalar query per head, one scalar key projection shared across heads, and the existing full-width value projection.

EVIDENCE: The 1,179-parameter design reached 100% accuracy with one-dimensional queries and keys; unlike unsuccessful reductions to positional biases, normalization, or residual projections, this isolates an eight-parameter reduction to source-token content addressing while retaining distinct head behavior through queries and relative biases.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.query_dim = 1
        self.query = nn.Linear(d_model, n_head, bias=False)
        self.key = nn.Linear(d_model, 1, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
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
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q + self.q_bias.expand(self.n_head)

        q = q.view(bsz, seqlen, self.n_head, self.query_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, 1, self.query_dim)
        k = k.expand(-1, -1, self.n_head, -1).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.query_dim)
>>>>>>> REPLACE