MECHANISM: Merge validated within-head key-bias groups

HYPOTHESIS: Sharing the scalar for first-head key coordinates 2/3 with second-head coordinates 0/2 will reduce the model from 1,612 to 1,611 parameters while retaining at least 99% accuracy, because it preserves both successful within-head tying patterns without further collapsing coordinates within either head.

INTENDED_EDIT: Store 15 QKV bias values, merge the two previously validated within-head key-bias groups, and shift the value-bias slices so their existing mapping remains unchanged.

EVIDENCE: Separate within-head ties reached 99.88% and 99.98%, while extending the second-head group to coordinate 3 fell to 19.76%; merging the two established groups tests a distinct reduction that leaves coordinate 3 independent.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 8))
=======
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 9))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + self.head_dim - 2],
                self.qkv.bias[d_model + self.head_dim - 3 : d_model + self.head_dim - 2],
                self.qkv.bias[d_model + self.head_dim - 2 : d_model + self.head_dim - 1],
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias[d_model + self.head_dim - 2 : 2 * d_model - 4],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 4 : 2 * d_model + self.head_dim - 7],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 4 : 2 * d_model - 2],
                self.qkv.bias[2 * d_model + self.head_dim - 7 :],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + self.head_dim - 2],
                self.qkv.bias[d_model + self.head_dim - 3 : d_model + self.head_dim - 2],
                self.qkv.bias[d_model + self.head_dim - 3 : d_model + self.head_dim - 2],
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias[d_model + self.head_dim - 3 : d_model + self.head_dim - 2],
                self.qkv.bias[d_model + self.head_dim - 2 : d_model + self.head_dim - 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 5 : 2 * d_model + self.head_dim - 8],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 5 : 2 * d_model - 3],
                self.qkv.bias[2 * d_model + self.head_dim - 8 :],
            )
        )
>>>>>>> REPLACE