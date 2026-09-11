MECHANISM: Incremental cross-head value-bias tying

HYPOTHESIS: Tying value-bias coordinates 3 and `head_dim + 3` will reduce the model from 1,615 to 1,614 parameters while retaining at least 99% accuracy, because the successful ties at coordinates 1 and 2 show that symmetric per-head value offsets can share learned degrees of freedom without the optimization failure caused by fixing additional offsets to zero.

INTENDED_EDIT: Store 18 QKV bias parameters and reconstruct all three nonzero value-bias coordinates of the second head from the corresponding learned coordinates of the first head, while retaining the validated fixed-zero key coordinate and balanced fixed-zero value coordinates.

EVIDENCE: Cross-head tying at value coordinate 1 achieved 99.86% accuracy at 1,616 parameters, and extending the tie to coordinate 2 achieved 99.75% at 1,615; tying coordinate 3 is the smallest direct continuation of this successful reduction path.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 5))
=======
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias[2 * d_model - 1 : 2 * d_model + 1],
                self.qkv.bias[2 * d_model + self.head_dim - 2 :],
=======
                self.qkv.bias[2 * d_model - 1 :],
>>>>>>> REPLACE