MECHANISM: Per-head gauge-fixed query bias

HYPOTHESIS: Constraining each query-bias vector to one learned scalar will reduce the model from 1,384 to 1,378 parameters while retaining at least 99% accuracy after 52,000 steps.

INTENDED_EDIT: Replace the eight-coordinate query bias with one broadcast scalar per attention head, preserving the existing fast linear and LayerNorm paths.

EVIDENCE: The current model reached 99.99% accuracy at 52,000 steps, while recent reductions that altered linear or LayerNorm kernels timed out; query/key coordinates have a rotational gauge, so fixing each head’s query-bias direction removes six redundant degrees of freedom without removing its learned content-independent attention term.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        # Query/key coordinates within each head have a rotational gauge.
        # Fix the query-bias direction and learn only its scalar magnitude.
        self.q_bias = nn.Parameter(torch.zeros(1, n_head, 1, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        q = q + self.q_bias
>>>>>>> REPLACE