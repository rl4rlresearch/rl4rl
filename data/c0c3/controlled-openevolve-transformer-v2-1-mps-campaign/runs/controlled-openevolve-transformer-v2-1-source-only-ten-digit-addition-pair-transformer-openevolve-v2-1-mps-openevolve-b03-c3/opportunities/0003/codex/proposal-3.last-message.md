MECHANISM: Remove softmax-invariant key bias

HYPOTHESIS: Retaining query and value biases while removing only the key bias will preserve at least 99% accuracy with 1,636 parameters, because a position-independent key bias adds the same scalar to every attention logit in a softmax row.

INTENDED_EDIT: Replace the fused QKV bias with separate learned query and value biases while leaving the key projection unbiased; all other architecture and training settings remain unchanged.

EVIDENCE: Removing both key and value biases reduced accuracy to 16.49%, so this narrower ablation preserves the potentially optimization-critical value bias while testing only the algebraically redundant key bias.

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