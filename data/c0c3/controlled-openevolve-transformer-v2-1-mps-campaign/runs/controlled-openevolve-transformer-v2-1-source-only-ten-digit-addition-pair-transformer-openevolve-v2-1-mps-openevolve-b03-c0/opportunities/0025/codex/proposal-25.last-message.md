MECHANISM: Incremental cross-head key-bias coordinate tying

HYPOTHESIS: Tying key-bias coordinates 2 and `head_dim + 2` will reduce the model from 1,614 to 1,613 parameters while retaining at least 99% accuracy, because the successful coordinate-1 key tie demonstrates that a learned symmetric constraint can preserve optimization despite fixed-zero key constraints failing.

INTENDED_EDIT: Store 17 QKV bias parameters and reconstruct the second head’s key-bias coordinates 1 and 2 from the corresponding learned coordinates of the first head, retaining the validated value-bias zeros and ties.

EVIDENCE: Tying key-bias coordinate 1 across heads achieved 99.91% accuracy at 1,614 parameters, whereas fixing an additional key coordinate at zero failed at 53.28%; extending the successful learned key tie by one coordinate is the smallest informative next reduction.

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
                self.qkv.bias[d_model : d_model + self.head_dim],
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias[d_model + self.head_dim : 2 * d_model - 2],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 2 : 2 * d_model + self.head_dim - 3],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 2 : 2 * d_model],
                self.qkv.bias[2 * d_model + self.head_dim - 3 :],
=======
                self.qkv.bias[d_model : d_model + self.head_dim],
                self.qkv.bias[d_model : d_model + 2],
                self.qkv.bias[d_model + self.head_dim : 2 * d_model - 3],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 3 : 2 * d_model + self.head_dim - 4],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 3 : 2 * d_model - 1],
                self.qkv.bias[2 * d_model + self.head_dim - 4 :],
>>>>>>> REPLACE