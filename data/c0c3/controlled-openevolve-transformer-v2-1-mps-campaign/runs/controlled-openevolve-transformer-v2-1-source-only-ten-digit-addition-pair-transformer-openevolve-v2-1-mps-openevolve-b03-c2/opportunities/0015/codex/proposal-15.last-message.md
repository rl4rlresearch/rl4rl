MECHANISM: Symmetric complete sharing of key-projection bias

HYPOTHESIS: Sharing all eight softmax-invariant key-bias coordinates as one learned scalar will produce a 1637-parameter model with at least 99% accuracy; unlike the failed 1637 design, it removes the asymmetric fixed-zero anchor.

INTENDED_EDIT: Replace the current one-zero/five-shared key-bias parameterization with a single learned value shared across every key-bias coordinate.

EVIDENCE: Six-coordinate sharing achieved 99.96% accuracy at 1638 parameters, while the asymmetric 1637 design with one fixed-zero coordinate and seven shared coordinates collapsed to 83.72%; fully symmetric sharing isolates whether that anchoring caused the optimization discontinuity.

<<<<<<< SEARCH
        # Remove five softmax-invariant key-bias degrees while preserving
        # fused-projection construction and a learned shared key-bias value.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 5))
=======
        # Share all softmax-invariant key-bias coordinates as one learned
        # scalar while preserving fused-projection construction.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model + 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1].expand(5),
                self.qkv.bias[d_model + 1 : 2 * d_model - 5],
                self.qkv.bias[2 * d_model - 5 :],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias[d_model : d_model + 1].expand(d_model),
                self.qkv.bias[d_model + 1 :],
            )
        )
>>>>>>> REPLACE