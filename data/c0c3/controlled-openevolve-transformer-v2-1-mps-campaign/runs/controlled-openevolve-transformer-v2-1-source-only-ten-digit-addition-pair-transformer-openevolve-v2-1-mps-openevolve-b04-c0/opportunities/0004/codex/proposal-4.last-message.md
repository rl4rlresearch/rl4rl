MECHANISM: Exact value/output-bias redundancy elimination

HYPOTHESIS: Removing the attention value bias will reduce parameters from 1636 to 1628 while retaining at least 99% accuracy, because softmax attention weights sum to one, making the shared value bias a constant that the existing output-projection bias can represent exactly.

INTENDED_EDIT: Delete the learned value bias and its addition while preserving the query bias, projection bias, LayerNorm parameters, feed-forward width, training, and decoding.

EVIDENCE: The isolated removal of the mathematically redundant key bias retained 99.95% accuracy at 1636 parameters; the failed 1596-parameter experiment confounded value-bias removal with disruptive non-affine LayerNorms, so isolating the other exact attention-bias redundancy is the most informative next ablation.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE