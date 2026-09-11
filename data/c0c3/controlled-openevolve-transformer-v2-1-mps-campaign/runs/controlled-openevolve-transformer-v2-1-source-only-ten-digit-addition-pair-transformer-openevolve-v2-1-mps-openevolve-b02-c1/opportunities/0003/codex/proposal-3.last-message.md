MECHANISM: Softmax-invariant key-bias elimination

HYPOTHESIS: Removing only the attention key bias will preserve at least 99% accuracy while reducing parameters from 1,644 to 1,636, because a shared key bias adds the same score offset to every visible key and therefore cancels exactly in the attention softmax.

INTENDED_EDIT: Replace the fused QKV bias with learned query and value biases, leaving keys unbiased while preserving the attention computation and residual width.

EVIDENCE: Compressing positional embeddings collapsed accuracy to 73.99%, and factorizing token embeddings collapsed it to 4.21%; this motivates retaining both learned representations and removing an analytically inactive attention parameter instead.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.qv_bias = nn.Parameter(torch.zeros(2, d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias, v_bias = self.qv_bias.unbind(0)
        q = q + q_bias
        v = v + v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE