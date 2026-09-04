MECHANISM: Eliminate redundant attention biases

HYPOTHESIS: Removing key and value biases while retaining the query bias will reduce parameters from 1644 to 1628 and preserve at least 99% accuracy because, with zero dropout, key bias cancels in the attention softmax and value bias is representable by the output-projection bias.

INTENDED_EDIT: Replace the combined QKV bias with a learned query-only bias.

EVIDENCE: Narrowing `d_ff` from 12 to 8 reduced accuracy to 71.49%, showing that nonlinear width is capacity-critical; the original `d_ff=12` model reached 99.96%, motivating an exact attention reparameterization instead.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Key bias cancels inside the softmax, and value bias can be
        # represented by the output-projection bias when dropout is zero.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE