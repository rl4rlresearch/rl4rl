MECHANISM: Function-preserving affine redundancy removal

HYPOTHESIS: Removing two redundant LayerNorm biases and replacing the attention QKV bias with only a query bias will reduce parameters from 1,644 to 1,612 while retaining at least 99% accuracy.

INTENDED_EDIT: Keep `d_ff=12`, remove both block LayerNorm biases, and retain only the functionally relevant query component of the QKV bias; constant key offsets cancel in softmax and constant value offsets are representable by the attention output bias.

EVIDENCE: The 1,644-parameter `d_ff=12` model achieved 99.96%, whereas widths 11 and 10 failed; this motivates preserving the successful hidden width and reducing affine parameters that do not expand the attention function class.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE