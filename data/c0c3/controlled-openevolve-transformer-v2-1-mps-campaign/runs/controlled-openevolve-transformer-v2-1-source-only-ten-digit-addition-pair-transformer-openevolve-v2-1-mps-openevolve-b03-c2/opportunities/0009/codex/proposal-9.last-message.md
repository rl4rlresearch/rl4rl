MECHANISM: Shared softmax-invariant key-bias coordinate

HYPOTHESIS: Tying a second key-bias coordinate to a surviving learned key bias will produce a 1642-parameter model with at least 99% accuracy, because the one-coordinate design achieved 99.88% and sharing avoids the additional fixed-zero coordinate used by the near-threshold 98.52% two-coordinate design.

INTENDED_EDIT: Preserve fused-QKV construction and initialization while replacing its 24-element bias with 22 learned values; reconstruct one zero key-bias coordinate and one tied key-bias coordinate during projection.

EVIDENCE: The qualified one-coordinate key-bias reduction reached 99.88% at 1643 parameters, while fixing two coordinates reached 98.52% at 1642; tying the second invariant coordinate is the smallest change that tests whether retaining a learned shared value improves the sensitive optimization trajectory.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Remove two redundant key-bias degrees while retaining one as a
        # shared learned value and preserving construction-time RNG use.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 2))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias[d_model : 2 * d_model - 2],
                self.qkv.bias[2 * d_model - 2 :],
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE