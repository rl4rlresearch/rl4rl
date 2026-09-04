MECHANISM: Remove softmax-invariant key bias

HYPOTHESIS: Eliminating the 8 key-bias parameters will reduce the model from 1,644 to 1,636 learned parameters while retaining at least 99% accuracy, because adding the same learned bias to every key shifts all attention logits for a query equally and therefore does not change softmax attention weights.

INTENDED_EDIT: Replace the combined QKV bias with separate learned query and value biases, leaving keys unbiased.

EVIDENCE: The 1,644-parameter design achieved 99.96% accuracy, whereas narrowing `d_ff` caused severe accuracy loss even at width 11; removing a mathematically redundant attention parameter is therefore better motivated than another capacity reduction.

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