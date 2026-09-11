MECHANISM: Remove functionally redundant key/value attention biases

HYPOTHESIS: Retaining the query bias and full 12-unit MLP while eliminating key and value biases will reduce parameters from 1644 to 1628 and preserve at least 99% accuracy.

INTENDED_EDIT: Make the packed QKV projection bias-free, add back only a learned query bias, and apply it after splitting Q, K, and V.

EVIDENCE: Reducing the MLP width to 8 cut accuracy to 44.84%, so this patch preserves the successful nonlinear capacity. Key bias cancels inside softmax, while the value-bias contribution can be represented by the existing output-projection bias.

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