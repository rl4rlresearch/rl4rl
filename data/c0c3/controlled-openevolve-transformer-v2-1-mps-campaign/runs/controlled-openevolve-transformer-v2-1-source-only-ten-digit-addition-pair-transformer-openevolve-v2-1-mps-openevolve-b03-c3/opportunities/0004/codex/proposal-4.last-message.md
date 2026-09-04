MECHANISM: Absorbable normalization and softmax-invariant biases

HYPOTHESIS: Removing the key-projection bias and the second LayerNorm bias will retain at least 99% accuracy with 1,628 parameters because key bias cancels under softmax and the LayerNorm offset can be absorbed by the existing MLP input bias.

INTENDED_EDIT: Preserve the width-12 MLP and query/value attention biases, while eliminating the redundant key bias and `ln2` bias.

EVIDENCE: The 1,636-parameter query/value-bias design achieved 99.97% accuracy, while narrowing the MLP failed; this motivates retaining nonlinear width and removing another structurally absorbable 8-parameter bias instead.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE