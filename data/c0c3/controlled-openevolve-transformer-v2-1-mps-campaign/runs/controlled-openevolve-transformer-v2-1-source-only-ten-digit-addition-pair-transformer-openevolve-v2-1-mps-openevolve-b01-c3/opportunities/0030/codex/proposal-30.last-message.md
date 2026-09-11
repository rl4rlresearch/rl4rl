MECHANISM: Shared value alphabet with independent attention routing

HYPOTHESIS: Sharing one learned value projection across both attention heads will reduce the verified fixed-bias design from 1,413 to 1,381 parameters while retaining at least 99% accuracy, because addition’s two operands use the same digit representation while independent query/key projections preserve each head’s load-bearing ability to route to different source positions.

INTENDED_EDIT: Replace per-head value projections with one learned four-dimensional value map shared by both heads, retain full-width independent query/key maps, and apply the verified fixed 0.02 shared query bias.

EVIDENCE: The 1,413-parameter fixed-bias design reached 99.96%, whereas reducing query/key routing width or positional capacity hurt accuracy. This suggests routing capacity is load-bearing, but does not establish that the two routed operands need separate value coordinate systems; a shared learned value alphabet directly tests that assumption while leaving routing intact.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(n_head, self.head_dim - 1))
        self.proj = MeanZeroLinear(d_model, d_model)
=======
        self.qk = nn.Linear(d_model, 2 * d_model, bias=False)
        self.value = nn.Linear(d_model, self.head_dim, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.proj = MeanZeroLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias = F.pad(self.q_bias, (0, 1)).reshape(d_model)
        q = q + q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qk = self.qk(x)
        q, k = qk.chunk(2, dim=-1)
        v = self.value(x)
        q_bias = F.pad(
            self.q_bias.expand(self.n_head, 1), (0, self.head_dim - 1)
        ).reshape(d_model)
        q = q + q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.unsqueeze(2).expand(
            bsz, seqlen, self.n_head, self.head_dim
        ).transpose(1, 2)
>>>>>>> REPLACE