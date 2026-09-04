MECHANISM: Exact softmax-invariant key-bias elimination

HYPOTHESIS: Removing only the attention key bias will reduce parameters from 1644 to 1636 while retaining at least 99% accuracy, because a shared key bias adds the same query-dependent constant to every unmasked attention logit and therefore cancels exactly in softmax.

INTENDED_EDIT: Replace the combined QKV bias with learned query and value biases, preserving the combined QKV weight, all LayerNorm affine parameters, and the original feed-forward width.

EVIDENCE: The 1644-parameter baseline reached 99.96%, whereas simultaneously removing LayerNorm affine parameters and key/value biases fell to 4.23%; this isolates the mathematically redundant key bias without repeating the disruptive LayerNorm or value-bias removals.

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