MECHANISM: Multi-query causal attention with a shared learned key/value head

HYPOTHESIS: Replacing the two independent key/value heads with one shared four-dimensional learned key/value head will retain at least 99% accuracy while reducing the current model from 1,623 to 1,555 parameters, because separate query heads still provide two operand-routing patterns while addition tokens can share a common content representation.

INTENDED_EDIT: Challenge the assumption that each query head requires its own key and value projections. Preserve the full eight-dimensional residual stream, tied lexical interface, two learned query heads, MLP, and qualified gauges, but broadcast one learned key/value head across both attention maps.

EVIDENCE: The full-width tied embedding is load-bearing—the rank-seven lexical factorization fell to 3.76%—and the gated-MLP alternative reached only 70.49%, so this patch preserves both. The 1,622-parameter four-gauge design reached 99.96%, while further gauge extensions became unstable, motivating a different reduction in attention redundancy that retains both independently learned routing queries.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # Keep a distinct learned query for each head, but share one learned
        # key/value representation across the query heads.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(self.head_dim))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        bsz, seqlen, d_model = x.shape
        q = self.q_proj(x) + self.q_bias
        k = self.k_proj(x)
        v = self.v_proj(x) + self.v_bias

        q = q.view(
            bsz, seqlen, self.n_head, self.head_dim
        ).transpose(1, 2)
        # Singleton head dimensions broadcast the shared learned key/value
        # stream across the independently parameterized query heads.
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
>>>>>>> REPLACE