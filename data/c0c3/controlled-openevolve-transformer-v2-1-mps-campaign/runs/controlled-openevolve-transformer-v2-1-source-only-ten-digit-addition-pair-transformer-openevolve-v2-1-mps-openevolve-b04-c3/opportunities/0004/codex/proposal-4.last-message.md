MECHANISM: RNG-preserving softmax-invariant key-bias removal

HYPOTHESIS: Removing the redundant key bias while preserving the baseline constructor RNG stream will achieve at least 99% accuracy with 1,636 parameters.

INTENDED_EDIT: Retain the original packed-linear initialization draw, discard its key-containing bias, learn separate zero-initialized query and value biases, and apply them explicitly.

EVIDENCE: The 1,644-parameter baseline achieved 99.96% accuracy. The previous key-bias removal unexpectedly reached only 71.66%; constructing the packed projection directly without bias changes subsequent seeded initialization, so preserving the baseline RNG stream isolates the mathematically invariant parameter removal.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Construct the original biased layer first so seeded initialization of
        # every later module retains the baseline RNG stream. The packed bias is
        # then replaced by only the functionally relevant query/value biases.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.qkv.register_parameter("bias", None)
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE