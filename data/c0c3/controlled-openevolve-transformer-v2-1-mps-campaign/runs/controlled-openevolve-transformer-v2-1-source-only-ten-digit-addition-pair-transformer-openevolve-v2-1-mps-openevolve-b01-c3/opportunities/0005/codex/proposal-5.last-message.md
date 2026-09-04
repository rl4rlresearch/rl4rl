MECHANISM: Downstream absorption of pre-normalization affine parameters

HYPOTHESIS: Combining the verified bias-minimal attention with affine-free pre-norm LayerNorms will reduce parameters from 1,628 to 1,596 while retaining at least 99% accuracy, because each removed LayerNorm scale can be absorbed into its following learned projection, while its shift is absorbed by the retained query, attention-output, and MLP input biases.

INTENDED_EDIT: Use bias-free packed QKV with only a learned query bias, and disable learned affine parameters in both transformer-block LayerNorms while preserving `d_ff=12` and both residual-output biases.

EVIDENCE: Reference Design 2 achieved 99.98% accuracy with 1,628 parameters after removing key and value biases; unlike the failed feed-forward-width and residual-output-bias reductions, this patch preserves those demonstrated capacities and removes downstream-reparameterizable normalization parameters.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
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

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE