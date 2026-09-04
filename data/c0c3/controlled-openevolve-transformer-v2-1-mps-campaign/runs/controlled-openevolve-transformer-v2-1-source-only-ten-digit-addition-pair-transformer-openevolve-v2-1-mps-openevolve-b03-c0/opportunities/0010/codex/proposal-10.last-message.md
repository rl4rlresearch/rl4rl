MECHANISM: Eliminate softmax-invariant key bias

HYPOTHESIS: Removing only the key-projection bias will reduce parameters from 1,620 to 1,612 while retaining at least 99% accuracy, because a position-independent key bias adds the same constant to every attention logit in a query row and is canceled by softmax.

INTENDED_EDIT: Replace the combined QKV bias with learned query and value biases, leaving the key projection bias-free while preserving all projection widths and other training settings.

EVIDENCE: Removing all 24 QKV biases failed at 49.86%, showing that indiscriminate bias removal is harmful; the key-bias component is uniquely removable without changing the attention function, making this a more targeted eight-parameter reduction.

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