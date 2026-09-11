MECHANISM: Within-head key-bias tying

HYPOTHESIS: Tying the second head’s key-bias coordinates 0 and 2 will reduce the model from 1,614 to 1,613 parameters while retaining at least 99% accuracy, because key biases are softmax-invariant and this avoids the cross-head coupling that failed for additional coordinate-aligned ties.

INTENDED_EDIT: Store 17 QKV bias parameters and reconstruct the second head’s key coordinate 0 from its independently stored coordinate 2, while preserving the validated key-coordinate-1 tie and all value-bias constraints.

EVIDENCE: The coordinate-1 cross-head key tie reached 99.91% at 1,614 parameters, while further cross-head ties at coordinates 2 and 3 failed; an untested within-head tie isolates whether cross-head coupling caused those optimization failures.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 6))
=======
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + self.head_dim],
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias[d_model + self.head_dim : 2 * d_model - 2],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 2 : 2 * d_model + self.head_dim - 3],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 2 : 2 * d_model],
                self.qkv.bias[2 * d_model + self.head_dim - 3 :],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + self.head_dim - 1],
                self.qkv.bias[d_model + self.head_dim - 1 : d_model + self.head_dim],
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias[d_model + self.head_dim - 1 : 2 * d_model - 3],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 3 : 2 * d_model + self.head_dim - 6],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 3 : 2 * d_model - 1],
                self.qkv.bias[2 * d_model + self.head_dim - 6 :],
            )
        )
>>>>>>> REPLACE