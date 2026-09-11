MECHANISM: Fixed-direction scalar addressing with tilted quartet bias

HYPOTHESIS: A 1,146-parameter transformer will achieve at least 99% accuracy because the 1,148-parameter fixed-direction scalar-addressing design reached 100%, while the current one-scalar tilted MLP bias independently reached 99.31%.

INTENDED_EDIT: Replace full-width QKV projections with the verified learned scalar gain on one normalized coordinate plus a full-width value projection, restore eleven-bin positional routing, and retain the current one-scalar tilted MLP bias.

EVIDENCE: The 1,148-parameter reference achieved 100% using fixed-direction scalar addressing; retaining the current verified tilted bias removes two additional parameters without modifying its successful attention mechanism.

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
        self.query_dim = 1
        self.address_scale = nn.Parameter(torch.full((1,), 0.02))
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
        address = self.address_scale * x[..., :1]
        v = self.value(x)

        q = address.unsqueeze(2).expand(-1, -1, self.n_head, -1)
        k = address.unsqueeze(2).expand(-1, -1, self.n_head, -1)
        q = q + self.q_bias.view(1, 1, 1, 1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
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