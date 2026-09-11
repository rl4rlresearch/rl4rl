MECHANISM: Remove functionally redundant attention biases

HYPOTHESIS: Retaining the width-12 MLP while removing key and value biases will preserve at least 99% accuracy with 1,628 parameters, because key bias cancels in softmax and value bias is representable by the output-projection bias.

INTENDED_EDIT: Replace the fused 24-parameter QKV bias with an 8-parameter learned query-only bias, saving 16 parameters without narrowing the model.

EVIDENCE: Reducing MLP width from 12 to 8 lowered accuracy from 99.96% to 62.9%, indicating nonlinear width should be preserved and structurally redundant parameters should be targeted instead.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
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