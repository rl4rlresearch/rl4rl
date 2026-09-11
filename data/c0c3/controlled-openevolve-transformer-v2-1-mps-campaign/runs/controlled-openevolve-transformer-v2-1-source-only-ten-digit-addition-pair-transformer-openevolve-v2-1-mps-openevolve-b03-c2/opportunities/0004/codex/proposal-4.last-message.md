MECHANISM: Eliminate softmax-invariant key bias

HYPOTHESIS: Removing only the 8-dimensional key-projection bias will reduce parameters from 1644 to 1636 while preserving at least 99% accuracy, because adding the same learned key bias to every position shifts all unmasked attention logits for a query by the same constant and therefore cannot change the softmax attention weights.

INTENDED_EDIT: Replace the fused QKV bias with separate learned query and value biases, leaving the key projection bias-free.

EVIDENCE: The verified 1644-parameter design achieved 99.96%, while every reduction of the feed-forward width failed; this motivates removing a functionally redundant attention parameter instead of further reducing demonstrated nonlinear capacity.

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
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE