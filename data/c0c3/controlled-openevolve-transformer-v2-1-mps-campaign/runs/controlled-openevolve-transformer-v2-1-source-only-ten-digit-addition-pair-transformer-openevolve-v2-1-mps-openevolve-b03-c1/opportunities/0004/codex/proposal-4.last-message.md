MECHANISM: RNG-aligned removal of softmax-invariant key bias

HYPOTHESIS: Removing the eight key-bias parameters while preserving the baseline constructor RNG stream and parameter ordering will retain at least 99% accuracy with 1,636 parameters.

INTENDED_EDIT: Retain the combined QKV weight, replace its 24-element bias with 16 learned query/value biases, and apply those biases explicitly.

EVIDENCE: The 1,644-parameter baseline reached 99.96%; the prior key-bias removal reached 95.93% despite key bias being mathematically irrelevant, motivating an initialization-aligned implementation that isolates the parameter removal from an RNG-trajectory change.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Construct the original affine first to preserve the baseline RNG stream,
        # then retain only the query and value portions of its learned bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = F.linear(x, self.qkv.weight)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias, v_bias = self.qkv.bias.chunk(2)
        q = q + q_bias
        v = v + v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE