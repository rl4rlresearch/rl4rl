MECHANISM: Alternative cross-head key-bias coordinate tying

HYPOTHESIS: Tying key-bias coordinate 3 across heads will reduce the model from 1,614 to 1,613 parameters while retaining at least 99% accuracy, because key biases are softmax-invariant and the successful coordinate-1 tie shows this redundancy can be removed without fixing another coordinate at zero.

INTENDED_EDIT: Store 17 learned QKV bias values and reconstruct the second head’s key-bias coordinate 3 from the first head’s coordinate 3, while retaining the validated coordinate-1 key tie and value-bias constraints.

EVIDENCE: The coordinate-1 key tie achieved 99.91% accuracy at 1,614 parameters, whereas tying coordinate 2 fell to 53.92%; testing coordinate 3 isolates whether that failure was coordinate-specific while continuing the mathematically redundant key-bias reduction path.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 6))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 7))
        self.proj = nn.Linear(d_model, d_model)
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
                self.qkv.bias[d_model : d_model + self.head_dim],
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias[d_model + self.head_dim : d_model + self.head_dim + 1],
                self.qkv.bias[d_model + self.head_dim - 2 : d_model + self.head_dim - 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 3 : 2 * d_model + self.head_dim - 4],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 3 : 2 * d_model - 1],
                self.qkv.bias[2 * d_model + self.head_dim - 4 :],
            )
        )
>>>>>>> REPLACE