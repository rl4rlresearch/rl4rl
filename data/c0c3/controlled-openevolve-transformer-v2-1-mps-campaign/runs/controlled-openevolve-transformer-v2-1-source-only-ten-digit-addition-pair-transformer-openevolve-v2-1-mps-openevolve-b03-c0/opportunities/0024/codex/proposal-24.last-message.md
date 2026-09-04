MECHANISM: Cross-head key-bias coordinate tying

HYPOTHESIS: Tying key-bias coordinates 1 and `head_dim + 1` will reduce the model from 1,615 to 1,614 parameters while retaining at least 99% accuracy, because key biases are softmax-invariant and a learned symmetric tie preserves more optimization freedom than fixing a second key coordinate at zero.

INTENDED_EDIT: Store 18 QKV bias parameters and reconstruct the second head’s key-bias coordinate 1 from the corresponding learned coordinate of the first head, retaining the validated value-bias zeros and ties.

EVIDENCE: Fixing a second key-bias coordinate at zero failed at 53.28%, while learned cross-head value-bias ties succeeded at 1,616 and 1,615 parameters; this motivates testing a learned cross-head tie on the mathematically redundant key bias rather than another fixed-zero constraint.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 5))
=======
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : 2 * d_model - 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 1 : 2 * d_model + self.head_dim - 2],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 1 : 2 * d_model + 1],
                self.qkv.bias[2 * d_model + self.head_dim - 2 :],
            )
        )
=======
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
>>>>>>> REPLACE