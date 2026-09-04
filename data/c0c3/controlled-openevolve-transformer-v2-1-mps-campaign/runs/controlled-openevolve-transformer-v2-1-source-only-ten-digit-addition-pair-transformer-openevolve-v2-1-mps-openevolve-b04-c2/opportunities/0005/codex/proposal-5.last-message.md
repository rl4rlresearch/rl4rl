MECHANISM: Initialization-preserving softmax-invariant key-bias elimination

HYPOTHESIS: Removing the eight-dimensional attention key bias while preserving the baseline RNG consumption and initial weights will reduce parameters from 1,644 to 1,636 and retain at least 99% accuracy.

INTENDED_EDIT: Construct the original biasful QKV layer to preserve initialization randomness, unregister its bias, and replace it with learned query/value biases only.

EVIDENCE: The previous key-bias removal reached only 64.55% despite the key bias being mathematically canceled by attention softmax, indicating that altered initialization—not lost representational capacity—likely caused the regression.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Construct with a full bias so subsequent parameters receive exactly the
        # same seeded initialization as the qualified design, then retain only
        # the query and value components that can affect attention outputs.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
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