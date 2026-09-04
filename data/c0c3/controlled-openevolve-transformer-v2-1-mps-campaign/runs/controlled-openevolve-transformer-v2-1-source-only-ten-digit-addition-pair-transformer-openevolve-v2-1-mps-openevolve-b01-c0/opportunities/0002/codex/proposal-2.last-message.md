MECHANISM: Eliminate softmax-invariant key bias

HYPOTHESIS: Removing only the attention key bias will reduce parameters from 1644 to 1636 without reducing accuracy below 99%, because a shared key bias adds the same scalar to every unmasked attention logit for each query and therefore cancels exactly under softmax.

INTENDED_EDIT: Replace the fused QKV linear bias with separate learned query and value biases while leaving keys unbiased.

EVIDENCE: Narrowing `d_ff` from 12 to 8 reduced parameters to 1576 but collapsed accuracy to 46.74%, showing that learned MLP capacity is essential; this patch instead removes eight attention parameters that are functionally redundant.

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