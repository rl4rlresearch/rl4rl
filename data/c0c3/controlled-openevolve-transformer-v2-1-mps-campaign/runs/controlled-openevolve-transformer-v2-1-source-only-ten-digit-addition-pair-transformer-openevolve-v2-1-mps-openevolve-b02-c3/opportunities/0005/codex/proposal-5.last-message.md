MECHANISM: Downstream absorption of the pre-MLP normalization bias

HYPOTHESIS: The query-only attention design with `ln2` bias removed will use 1620 parameters and retain at least 99% accuracy because `fc1.bias` can absorb the LayerNorm offset while the learned LayerNorm scale remains available.

INTENDED_EDIT: Apply the proven query-only QKV bias parameterization and remove only the eight-parameter bias from the pre-MLP LayerNorm.

EVIDENCE: Query-only attention reached 100% accuracy with 1628 parameters. Removing all `ln2` affine parameters failed at 59.17%, motivating the narrower test that preserves its potentially capacity-critical learned scale.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Key bias cancels inside the softmax, and value bias can be
        # represented by the output-projection bias when dropout is zero.
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
        # Its additive offset is representable by the following fc1 bias.
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE