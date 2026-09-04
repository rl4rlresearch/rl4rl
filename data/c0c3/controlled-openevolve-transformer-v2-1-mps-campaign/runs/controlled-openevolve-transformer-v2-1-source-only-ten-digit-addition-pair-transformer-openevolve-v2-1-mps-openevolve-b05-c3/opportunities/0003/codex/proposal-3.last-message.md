MECHANISM: Remove softmax-invariant key bias

HYPOTHESIS: Keeping d_ff=12 while removing the key-projection bias will reduce parameters from 1644 to 1636 and retain at least 99% accuracy because a shared key bias adds the same constant to every unmasked attention logit and therefore cancels under softmax.

INTENDED_EDIT: Replace the fused QKV bias with learned query and value biases only; leave architecture, training, and decoding otherwise unchanged.

EVIDENCE: The d_ff=12 model achieved 99.96% accuracy, while d_ff=11 fell to 73.41%, motivating retention of the proven MLP capacity and removal of a functionally redundant attention parameter instead.

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