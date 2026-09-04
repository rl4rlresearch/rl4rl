MECHANISM: Gauge-free direct attention-head operators

HYPOTHESIS: Folding each head’s Q/K and V/output factorizations into unrestricted quotient-space operators, on top of the qualified 1,279-parameter design, will produce a 1,257-parameter transformer with at least 99% accuracy because the direct operators contain every function representable by the removed rank-four factorizations.

INTENDED_EDIT: Adopt the verified quotient-input MLP and five-direction pre-MLP bias, then replace factorized QKV/output attention with learned per-head score and value-output matrices while preserving causal masking, relative biases, dropout, and generic decoding.

EVIDENCE: The 1,279-parameter quotient-input design achieved 99.97% accuracy. Its attention still represents each head’s score as `WqᵀWk` and value/output map as `WoWv`; learning those products directly removes 22 factorization-gauge parameters without narrowing their representable operators.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = QuotientInputLinear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.quotient_dim = d_model - 1

        basis = torch.zeros(d_model, self.quotient_dim)
        for col in range(self.quotient_dim):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

        factor_std = math.sqrt(self.head_dim) * (0.02 ** 2)
        self.score_weight = nn.Parameter(
            torch.empty(n_head, self.quotient_dim, self.quotient_dim)
        )
        self.key_score_bias = nn.Parameter(
            torch.zeros(n_head, self.quotient_dim)
        )
        self.value_weight = nn.Parameter(
            torch.empty(n_head, self.quotient_dim, self.quotient_dim)
        )
        self.output_bias = nn.Parameter(torch.zeros(self.quotient_dim))
        nn.init.normal_(self.score_weight, mean=0.0, std=factor_std)
        nn.init.normal_(self.value_weight, mean=0.0, std=factor_std)

        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
=======
        _, seqlen, _ = x.shape
        coordinates = F.linear(x, self.basis.transpose(0, 1))
        att = torch.einsum(
            "btf,hfg,bsg->bhts",
            coordinates,
            self.score_weight,
            coordinates,
        )
        key_score = torch.einsum(
            "hf,bsf->bhs", self.key_score_bias, coordinates
        )
        att = (att + key_score.unsqueeze(2)) / math.sqrt(self.head_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
        return y
=======
        context = torch.einsum("bhts,bsf->bhtf", att, coordinates)
        y = torch.einsum("bhtf,hgf->btg", context, self.value_weight)
        y = F.linear(y + self.output_bias, self.basis)
        y = self.resid_drop(y)
        return y
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = QuotientInputLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = BiasOnlyLayerNorm(cfg.d_model)
=======
        self.ln2 = GaugeFixedLayerNorm(cfg.d_model)
>>>>>>> REPLACE