MECHANISM: Downstream absorption of pre-MLP normalization affine parameters

HYPOTHESIS: Retaining the proven query-only attention bias while making the pre-MLP LayerNorm non-affine will reduce parameters from 1628 to 1612 and maintain at least 99% accuracy, because its scale and bias can be learned equivalently by the immediately following `fc1` weight and bias.

INTENDED_EDIT: Use bias-free QKV with an explicit learned query bias, and remove the redundant 16-parameter affine transform from `ln2`.

EVIDENCE: Query-only attention bias achieved 100% accuracy with 1628 parameters, while narrowing `d_ff` caused accuracy to fall to 71.49%; this motivates preserving MLP width and removing parameters that are functionally absorbable by its first projection.

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
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE