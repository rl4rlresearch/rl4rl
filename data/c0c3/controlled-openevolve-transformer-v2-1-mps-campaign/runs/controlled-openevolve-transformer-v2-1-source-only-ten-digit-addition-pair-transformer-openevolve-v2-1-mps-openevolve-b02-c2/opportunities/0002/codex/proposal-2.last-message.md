MECHANISM: Softmax-invariant attention key-bias removal

HYPOTHESIS: Removing only the functionally redundant key bias will reduce the model from 1,644 to 1,636 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Replace the combined QKV bias with learned query and value biases, leaving keys unbiased.

EVIDENCE: The 1,644-parameter design reached 99.96%, while narrowing `d_ff` to 10 collapsed to 9.07%; this motivates preserving representational width and removing a key bias whose contribution cancels exactly inside attention softmax.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias, v_bias = self.qv_bias.chunk(2)
        q = q + q_bias
        v = v + v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE