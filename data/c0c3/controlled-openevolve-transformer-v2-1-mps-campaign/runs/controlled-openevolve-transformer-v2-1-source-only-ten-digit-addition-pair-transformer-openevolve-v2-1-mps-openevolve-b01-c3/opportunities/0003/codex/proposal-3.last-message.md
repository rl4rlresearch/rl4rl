MECHANISM: Softmax-invariant key-bias elimination

HYPOTHESIS: Removing only the attention key bias will reduce parameters from 1,644 to 1,636 while retaining at least 99% accuracy, because a position-independent key bias adds the same value to every unmasked attention logit for a query and therefore cancels exactly in softmax.

INTENDED_EDIT: Replace the packed QKV bias with learned query and value biases, preserving all functional attention biases, `d_ff=12`, and both residual-output biases.

EVIDENCE: The 1,644-parameter design reached 99.96% accuracy, while reducing nonlinear width or removing residual-output biases caused large regressions; this patch preserves those demonstrated capacities and removes only a mathematically non-identifiable parameter vector.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias, v_bias = self.qv_bias.chunk(2)
        q = q + q_bias
        v = v + v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE