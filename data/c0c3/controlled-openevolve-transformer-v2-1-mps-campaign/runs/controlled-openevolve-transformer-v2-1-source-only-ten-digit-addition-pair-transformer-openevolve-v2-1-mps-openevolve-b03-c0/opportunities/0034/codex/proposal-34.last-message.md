MECHANISM: Same-head query/key bias reuse

HYPOTHESIS: Sharing the remaining second-head key-bias coordinate with its corresponding freely learned query-bias coordinate will reduce the model from 1,611 to 1,610 parameters while retaining at least 99% accuracy, because a position-independent key offset cancels in the attention softmax without constraining the query path.

INTENDED_EDIT: Store 14 QKV bias parameters, reconstruct the second head’s final key bias from its final query-bias coordinate, and shift the value-bias slices to preserve their validated mapping.

EVIDENCE: Learned key-bias sharing reached 99.96% at 1,611 parameters, while merging the remaining key coordinate into the existing coordinate-2 group failed at 19.76%; query/key reuse tests a distinct reduction without further collapsing either successful key group or any value-bias degree of freedom.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 9))
=======
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 10))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias[d_model + self.head_dim - 2 : d_model + self.head_dim - 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 5 : 2 * d_model + self.head_dim - 8],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 5 : 2 * d_model - 3],
                self.qkv.bias[2 * d_model + self.head_dim - 8 :],
=======
                self.qkv.bias[d_model - 1 : d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 6 : 2 * d_model + self.head_dim - 9],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 6 : 2 * d_model - 4],
                self.qkv.bias[2 * d_model + self.head_dim - 9 :],
>>>>>>> REPLACE