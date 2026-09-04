MECHANISM: Full-width tied query-key metric attention

HYPOTHESIS: Sharing each head’s full-dimensional query and key representation will reduce the model from 1,372 to 1,316 parameters while retaining at least 99% accuracy within 50,000 steps, because the causal mask supplies direction while learned positional representations can encode the required symmetric routing affinities.

INTENDED_EDIT: Replace the independent query and key projections with one shared full-width projection, retain a separate value projection and the learned query offset, and continue using a single fused linear operation.

EVIDENCE: The 1,372-parameter model achieved 100% accuracy, and a rank-six positional representation previously achieved 99.98%, indicating routing tolerates constrained address geometry. The earlier query/key bottleneck provided no contrary accuracy evidence because it timed out; this version preserves all four routing dimensions per head, batch size 512, and the optimized fused-linear path.

<<<<<<< SEARCH
        self.qkv = MeanZeroInputLinear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = MeanZeroInputLinear(d_model, 2 * d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        qk, v = qkv.chunk(2, dim=-1)

        q = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE