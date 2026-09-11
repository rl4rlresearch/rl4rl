MECHANISM: Exact softmax-invariant key-bias removal

HYPOTHESIS: Removing only the 8 key-projection bias parameters will reduce the model from 1644 to 1636 parameters while retaining at least 99% accuracy, because a position-independent key bias adds the same attention-logit offset for every attended key and cancels exactly under softmax.

INTENDED_EDIT: Replace the packed QKV bias with separate learned query and value biases, leaving the key projection bias-free while preserving all other architecture and training settings.

EVIDENCE: Removing both key and value biases reduced accuracy to 52.73%, so value-bias removal may have disrupted optimization; isolating the mathematically redundant key bias tests a smaller, function-preserving reduction.

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