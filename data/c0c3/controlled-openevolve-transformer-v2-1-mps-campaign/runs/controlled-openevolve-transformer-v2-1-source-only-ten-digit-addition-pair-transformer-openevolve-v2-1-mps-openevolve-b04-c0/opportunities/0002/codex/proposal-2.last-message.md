MECHANISM: Gauge-free pre-normalization with bias-minimal attention

HYPOTHESIS: Removing reparameterizable pre-LayerNorm affine parameters and redundant key/value biases will reduce parameters from 1644 to 1596 while retaining at least 99% accuracy, because it preserves width and representational capacity unlike the failed `d_ff=8` ablation.

INTENDED_EDIT: Make both pre-LayerNorms non-affine and replace the combined QKV projection with separate projections that retain only the functionally relevant query bias.

EVIDENCE: The 1644-parameter `d_ff=12` model achieved 99.96%, while narrowing `d_ff` to 8 produced only 69.64%; this motivates preserving the nonlinear width and instead removing parameterization redundancies.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
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