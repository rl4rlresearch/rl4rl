MECHANISM: Incremental sharing of a softmax-invariant key bias

HYPOTHESIS: Extending the verified shared key-bias parameterization by one coordinate will produce a 1638-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reduce the fused QKV bias from 23 to 18 learned values, reconstructing one zero key-bias coordinate and six key coordinates from one shared parameter.

EVIDENCE: The 1639-parameter design achieved 99.55% accuracy after the same incremental sharing mechanism succeeded at 1642, 1641, and 1640 parameters, making one further shared coordinate the most informative next reduction.

<<<<<<< SEARCH
        # Keep the fused projection and its construction-time RNG consumption,
        # while fixing one softmax-invariant key-bias coordinate at zero.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 1))
=======
        # Remove six softmax-invariant key-bias degrees while preserving
        # fused-projection construction and a learned shared key-bias value.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model:],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1].expand(6),
                self.qkv.bias[d_model + 1 : 2 * d_model - 6],
                self.qkv.bias[2 * d_model - 6 :],
            )
        )
>>>>>>> REPLACE