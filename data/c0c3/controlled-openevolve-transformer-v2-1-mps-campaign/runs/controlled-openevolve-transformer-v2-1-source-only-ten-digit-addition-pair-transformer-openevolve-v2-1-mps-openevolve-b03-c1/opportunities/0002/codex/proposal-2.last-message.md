MECHANISM: Remove softmax-invariant key bias

HYPOTHESIS: Removing only the key-projection bias will reduce parameters from 1,644 to 1,636 while retaining at least 99% accuracy because a position-independent key bias adds the same constant to every unmasked attention score and therefore cannot change softmax attention weights.

INTENDED_EDIT: Replace the combined QKV bias with learned query and value biases, preserving the full d_ff=12 architecture while eliminating the redundant eight-parameter key bias.

EVIDENCE: The 1,644-parameter d_ff=12 model achieved 99.96% accuracy, whereas narrowing d_ff to 8 reduced accuracy to 71%; this motivates preserving feed-forward capacity and removing a mathematically redundant attention parameter instead.

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