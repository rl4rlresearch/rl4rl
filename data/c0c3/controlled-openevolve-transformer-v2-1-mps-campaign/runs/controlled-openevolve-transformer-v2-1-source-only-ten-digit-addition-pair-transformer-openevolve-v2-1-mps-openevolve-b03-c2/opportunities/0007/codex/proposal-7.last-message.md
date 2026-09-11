MECHANISM: Two-coordinate softmax-invariant key-bias removal

HYPOTHESIS: Fixing two key-bias coordinates at zero will reduce the model from 1644 to 1642 parameters while retaining at least 99% accuracy, because the qualified one-coordinate design achieved 99.88% and every constant key-bias coordinate is canceled by attention softmax.

INTENDED_EDIT: Preserve the fused QKV layer and constructor RNG consumption, replace its 24-element bias with 22 learned elements, and reconstruct two fixed-zero key-bias coordinates during projection.

EVIDENCE: The 1643-parameter one-coordinate design achieved 99.88%, while removing all eight key-bias coordinates failed; removing one additional invariant coordinate is the smallest informative continuation.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Preserve fused-projection construction while fixing two
        # softmax-invariant key-bias coordinates at zero.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 2))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        bsz, seqlen, d_model = x.shape
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(2),
                self.qkv.bias[d_model:],
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE