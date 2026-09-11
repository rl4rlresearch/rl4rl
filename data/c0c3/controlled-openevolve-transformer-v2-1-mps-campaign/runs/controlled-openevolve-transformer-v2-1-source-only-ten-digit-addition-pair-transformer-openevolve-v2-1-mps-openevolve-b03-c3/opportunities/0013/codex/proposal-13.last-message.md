MECHANISM: Affine-free pre-MLP normalization

HYPOTHESIS: Removing the remaining `ln2` scale parameters from the verified 1,628-parameter design will retain at least 99% accuracy with 1,620 parameters, because `ln2` initially applies unit scaling and any learned per-channel scaling can be absorbed by the following MLP input weights.

INTENDED_EDIT: Remove the softmax-invariant key bias, retain learned query/value biases, and make the second LayerNorm entirely non-affine while preserving the width-12 MLP and all training settings.

EVIDENCE: The non-key-biased, `ln2`-bias-free design achieved 99.95% accuracy with 1,628 parameters; unlike failed width reduction, removing `ln2` scale preserves all nonlinear channels and leaves initialization functionally unchanged.

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

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE