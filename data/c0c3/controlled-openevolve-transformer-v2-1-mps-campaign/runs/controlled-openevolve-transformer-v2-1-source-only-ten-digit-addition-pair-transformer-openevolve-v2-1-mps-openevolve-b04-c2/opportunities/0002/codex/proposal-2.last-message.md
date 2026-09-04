MECHANISM: Softmax-invariant key-bias elimination

HYPOTHESIS: Removing only the attention key bias will reduce parameters from 1,644 to 1,636 while preserving at least 99% accuracy because a shared key bias adds the same constant to every visible attention logit and cancels under softmax.

INTENDED_EDIT: Replace the combined QKV bias with learned query and value biases, leaving the key projection unbiased while retaining every other projection bias.

EVIDENCE: Removing all 52 attention and MLP biases reduced accuracy to 72.22%, so those biases should not be removed wholesale; this patch targets only the eight-dimensional key bias whose contribution is mathematically invariant under attention softmax.

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