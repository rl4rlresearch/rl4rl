MECHANISM: Single-coordinate softmax-invariant key-bias elimination

HYPOTHESIS: Removing one individually irrelevant key-bias coordinate will retain at least 99% accuracy while reducing learned parameters from 1,644 to 1,643.

INTENDED_EDIT: Preserve the original linear-constructor RNG draw and fused QKV computation, but store 23 packed bias values and reconstruct one key-bias coordinate as zero.

EVIDENCE: The baseline achieved 99.96% accuracy, while removing all eight key-bias coordinates failed despite their mathematical redundancy; eliminating only one coordinate is the smallest perturbation and avoids the quotient-coordinate changes introduced by the failed MLP-bias reduction.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Preserve the constructor's original RNG consumption, then remove one
        # key-bias coordinate that cancels identically inside attention softmax.
        self.qkv.register_parameter("bias", None)
        self.qkv.register_parameter(
            "reduced_bias", nn.Parameter(torch.zeros(3 * d_model - 1))
        )
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        reduced_bias = self.qkv.reduced_bias
        qkv_bias = torch.cat(
            (
                reduced_bias[:d_model],
                reduced_bias.new_zeros(1),
                reduced_bias[d_model:],
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE