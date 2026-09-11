MECHANISM: Incremental sharing of a softmax-invariant key bias

HYPOTHESIS: Extending the verified shared key-bias parameterization by one coordinate will produce a 1639-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reduce the fused QKV bias from 22 to 19 learned values, reconstructing one zero key-bias coordinate and five key coordinates from one shared parameter.

EVIDENCE: The 1640-parameter design achieved 99.98% accuracy after the same sharing mechanism succeeded at 1642 and 1641 parameters, motivating the next one-parameter reduction.

<<<<<<< SEARCH
        # Remove two redundant key-bias degrees while retaining one as a
        # shared learned value and preserving construction-time RNG use.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 2))
=======
        # Remove five softmax-invariant key-bias degrees while preserving
        # fused-projection construction and a learned shared key-bias value.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias[d_model : 2 * d_model - 2],
                self.qkv.bias[2 * d_model - 2 :],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1].expand(5),
                self.qkv.bias[d_model + 1 : 2 * d_model - 5],
                self.qkv.bias[2 * d_model - 5 :],
            )
        )
>>>>>>> REPLACE