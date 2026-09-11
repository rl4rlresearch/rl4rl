MECHANISM: Pre-normalization bias absorption

HYPOTHESIS: Retaining only the effective query bias and removing both pre-sublayer LayerNorm biases will reduce the model to 1,612 parameters while maintaining at least 99% accuracy, because those normalization offsets are representable by the downstream query, attention-output, and MLP biases.

INTENDED_EDIT: Apply the verified query-only fused-QKV reparameterization and disable the biases in `ln1` and `ln2`.

EVIDENCE: The 1,628-parameter query-only-bias design achieved 99.97% accuracy; its margin motivates removing 16 additional parameters that are initialized to zero and functionally absorbable by existing downstream biases.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Preserve fused projection construction while retaining only query bias.
        # Key bias cancels in softmax and value bias is absorbed by proj.bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        q_bias = self.qkv.bias
        fused_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), q_bias.new_zeros(d_model))
        )
        qkv = F.linear(x, self.qkv.weight, fused_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        # Pre-sublayer normalization offsets are absorbed by downstream biases.
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE