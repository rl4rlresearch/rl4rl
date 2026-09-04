MECHANISM: Incremental second-head key-bias tying

HYPOTHESIS: Tying the second head’s key-bias coordinate 3 to coordinate 2 will reduce the model from 1,612 to 1,611 parameters while retaining at least 99% accuracy, because the same within-head coordinate-2/3 tie already succeeded in the first head and the second head’s coordinate-0/2 tie also succeeded.

INTENDED_EDIT: Store 15 QKV bias values, reconstruct the second head’s key coordinates 0, 2, and 3 from one learned scalar, and shift the downstream value-bias slices to preserve their current mapping.

EVIDENCE: Within-head key tying succeeded twice: tying second-head coordinates 0 and 2 achieved 99.88% at 1,613 parameters, and tying first-head coordinates 2 and 3 achieved 99.98% at 1,612; applying the validated coordinate-2/3 pattern to the second head is the smallest direct continuation.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 8))
=======
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 9))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias[d_model + self.head_dim - 2 : 2 * d_model - 4],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 4 : 2 * d_model + self.head_dim - 7],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 4 : 2 * d_model - 2],
                self.qkv.bias[2 * d_model + self.head_dim - 7 :],
=======
                self.qkv.bias[d_model + self.head_dim - 2 : d_model + self.head_dim - 1],
                self.qkv.bias[d_model + self.head_dim - 2 : d_model + self.head_dim - 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model + self.head_dim - 1 : d_model + self.head_dim],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model + self.head_dim - 1 : d_model + self.head_dim + 1],
                self.qkv.bias[d_model + self.head_dim :],
>>>>>>> REPLACE