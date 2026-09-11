MECHANISM: QKV null-coordinate recycling

HYPOTHESIS: Reusing one QKV row’s LayerNorm-null common component as the shared query-bias scalar will produce a 1,526-parameter model with at least 99% accuracy while preserving the function class of the verified 1,527-parameter design.

INTENDED_EDIT: Remove the standalone query-bias parameters, derive the shared bias from a normalized sum of one QKV row, and center that row after initialization so the bias retains its verified zero initialization.

EVIDENCE: The shared learned query bias reached 99.8% with 1,527 parameters, whereas deleting a QKV null coordinate collapsed to 29.57%; recycling that coordinate keeps it trainable and preserves the dense QKV parameterization instead of asymmetrically removing it.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(n_head, 1))
        self.proj = MeanZeroLinear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = MeanZeroLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias = F.pad(self.q_bias, (0, self.head_dim - 1)).reshape(d_model)
        q = q + q_bias
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        shared_q_bias = self.qkv.weight[0].sum() / math.sqrt(d_model)
        q_bias = F.pad(
            shared_q_bias.expand(self.n_head, 1), (0, self.head_dim - 1)
        ).reshape(d_model)
        q = q + q_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
=======
        self.apply(self._init_weights)

        # The common component of this row is invisible to mean-zero ln1
        # inputs, so initialize its recycled query-bias coordinate at zero.
        with torch.no_grad():
            for block in self.blocks:
                carrier = block.attn.qkv.weight[0]
                carrier.sub_(carrier.mean())
>>>>>>> REPLACE