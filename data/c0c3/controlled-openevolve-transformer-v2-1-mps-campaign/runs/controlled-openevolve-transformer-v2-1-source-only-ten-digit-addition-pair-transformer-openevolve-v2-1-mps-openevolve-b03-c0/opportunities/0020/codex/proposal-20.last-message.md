MECHANISM: Balanced per-head key-bias elimination

HYPOTHESIS: Removing key-bias coordinate 0 from each attention head will reduce the model from 1,617 to 1,616 parameters while retaining at least 99% accuracy, because key biases are softmax-invariant and distributing constraints across heads previously succeeded for value biases where adjacent same-head removal failed.

INTENDED_EDIT: Store 20 QKV bias values and reconstruct fixed-zero key coordinates 0 and `head_dim`, plus the already validated fixed-zero value coordinates 0 and `head_dim`, while preserving layer-construction RNG trajectory.

EVIDENCE: Concentrating two removed value-bias coordinates in the first head reached only 96.11%, whereas distributing them across the two heads reached 99.97% at 1,617 parameters; this motivates applying the same balanced-head strategy to the next key-bias reduction.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 3))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 4))
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
                self.qkv.bias[2 * d_model + self.head_dim - 2 :],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + self.head_dim - 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model + self.head_dim - 1 : 2 * d_model - 2],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 2 : 2 * d_model + self.head_dim - 3],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model + self.head_dim - 3 :],
            )
        )
>>>>>>> REPLACE