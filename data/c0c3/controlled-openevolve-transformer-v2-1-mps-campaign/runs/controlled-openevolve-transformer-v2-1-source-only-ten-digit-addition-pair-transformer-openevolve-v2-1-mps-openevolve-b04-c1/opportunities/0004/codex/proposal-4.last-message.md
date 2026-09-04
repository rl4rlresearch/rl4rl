MECHANISM: Remove softmax-invariant key-projection bias

HYPOTHESIS: Eliminating only the 8 key-bias parameters will reduce the model from 1644 to 1636 parameters while retaining at least 99% accuracy, because adding the same learned key bias to every attended position changes each query’s attention logits by a position-independent constant that softmax cancels exactly.

INTENDED_EDIT: Replace the fused 24-element QKV bias with separate learned query and value biases, preserving the original query/value computations while omitting the functionally redundant key bias.

EVIDENCE: The 1644-parameter baseline reached 99.96% accuracy, while removing all 52 linear biases collapsed accuracy to 3.99%; this motivates a targeted removal limited to the key-bias component whose effect on causal attention weights is mathematically invariant.

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