MECHANISM: Incremental sharing of a softmax-invariant key-bias coordinate

HYPOTHESIS: Tying one additional key-bias coordinate to the existing shared value will produce a 1640-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reduce the fused QKV bias from 21 to 20 learned values, reconstructing one zero key-bias coordinate and four key coordinates from one shared parameter.

EVIDENCE: The same incremental sharing mechanism reduced the qualified model from 1642 parameters at 99.91% accuracy to 1641 parameters at 99.74%, leaving sufficient accuracy headroom for the next one-parameter probe.

<<<<<<< SEARCH
        # Remove three softmax-invariant key-bias degrees while preserving
        # fused-projection construction and a learned shared key-bias value.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 3))
=======
        # Remove four softmax-invariant key-bias degrees while preserving
        # fused-projection construction and a learned shared key-bias value.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias[d_model : d_model + 1].expand(3),
                self.qkv.bias[d_model + 1 : 2 * d_model - 3],
                self.qkv.bias[2 * d_model - 3 :],
=======
                self.qkv.bias[d_model : d_model + 1].expand(4),
                self.qkv.bias[d_model + 1 : 2 * d_model - 4],
                self.qkv.bias[2 * d_model - 4 :],
>>>>>>> REPLACE