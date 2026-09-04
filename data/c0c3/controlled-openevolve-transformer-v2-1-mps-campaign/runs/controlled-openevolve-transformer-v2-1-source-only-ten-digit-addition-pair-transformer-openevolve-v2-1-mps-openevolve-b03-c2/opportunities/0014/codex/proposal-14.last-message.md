MECHANISM: Complete sharing of softmax-invariant key bias

HYPOTHESIS: Tying the final independent key-bias coordinate to the existing shared value will produce a 1637-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reduce the fused QKV bias from 18 to 17 learned values, reconstructing one fixed-zero key coordinate and all seven remaining key coordinates from one shared parameter.

EVIDENCE: The current 1638-parameter design achieved 99.96% accuracy after incremental key-bias sharing also qualified at 1642, 1641, 1640, and 1639 parameters, providing strong evidence for the final one-parameter continuation.

<<<<<<< SEARCH
        # Remove six softmax-invariant key-bias degrees while preserving
        # fused-projection construction and a learned shared key-bias value.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 6))
=======
        # Remove seven softmax-invariant key-bias degrees while preserving
        # fused-projection construction and a learned shared key-bias value.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias[d_model : d_model + 1].expand(6),
                self.qkv.bias[d_model + 1 : 2 * d_model - 6],
                self.qkv.bias[2 * d_model - 6 :],
=======
                self.qkv.bias[d_model : d_model + 1].expand(7),
                self.qkv.bias[2 * d_model - 7 :],
>>>>>>> REPLACE