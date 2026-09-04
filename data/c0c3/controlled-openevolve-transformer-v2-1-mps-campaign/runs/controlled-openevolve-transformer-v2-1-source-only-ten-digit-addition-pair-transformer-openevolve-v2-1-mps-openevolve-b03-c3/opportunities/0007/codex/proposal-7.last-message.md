MECHANISM: Value-bias absorption of the attention output offset

HYPOTHESIS: Removing the key bias, `ln2` bias, and attention output-projection bias will retain at least 99% accuracy with 1,620 parameters because the retained value bias supplies the projection’s constant offset while preserving the optimization-critical value-bias pathway.

INTENDED_EDIT: Use unbiased fused QKV weights with learned query/value biases, remove the `ln2` bias, and make the attention output projection unbiased.

EVIDENCE: The 1,628-parameter design removing key and `ln2` biases achieved 99.95% accuracy, while removing the value bias collapsed accuracy to 16.49%; this motivates retaining value bias and testing the output bias that it can functionally absorb.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model, bias=False)
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