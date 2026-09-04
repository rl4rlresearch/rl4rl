MECHANISM: Eliminate softmax-invariant key bias

HYPOTHESIS: Removing only the 8-dimensional key bias will reduce parameters from 1,644 to 1,636 while retaining at least 99% accuracy, because adding the same learned key bias to every position shifts all unmasked attention logits by an equal amount and therefore cannot change the attention probabilities.

INTENDED_EDIT: Replace the combined QKV bias with separate learned query and value biases, leaving the key projection unbiased while preserving `d_ff=12`.

EVIDENCE: The 1,644-parameter `d_ff=12` model reached 99.96%, while reducing `d_ff` to 11 collapsed accuracy to 74.06%; this motivates preserving demonstrated feed-forward capacity and removing a mathematically redundant attention parameter instead.

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