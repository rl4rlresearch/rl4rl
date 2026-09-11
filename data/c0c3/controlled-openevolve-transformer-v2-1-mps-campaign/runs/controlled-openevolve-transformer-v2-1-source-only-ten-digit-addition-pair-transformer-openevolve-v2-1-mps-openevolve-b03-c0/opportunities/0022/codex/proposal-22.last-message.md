MECHANISM: Incremental cross-head value-bias tying

HYPOTHESIS: Tying value-bias coordinates 2 and `head_dim + 2` will reduce the model from 1,616 to 1,615 parameters while retaining at least 99% accuracy, because the successful cross-head tie at coordinate 1 shows that symmetric per-head value offsets can share a learned degree of freedom without imposing a fixed-zero constraint.

INTENDED_EDIT: Store 19 QKV bias parameters and reconstruct the second head’s value-bias coordinates 1 and 2 from the corresponding learned coordinates of the first head.

EVIDENCE: The current symmetric cross-head value-bias tie achieved 99.86% accuracy at 1,616 parameters, whereas fixing an additional value coordinate at zero achieved only 96.11%; extending the successful learned tie by one coordinate is the smallest informative reduction.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 4))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias[2 * d_model - 1 : 2 * d_model],
                self.qkv.bias[2 * d_model + self.head_dim - 2 :],
=======
                self.qkv.bias[2 * d_model - 1 : 2 * d_model + 1],
                self.qkv.bias[2 * d_model + self.head_dim - 2 :],
>>>>>>> REPLACE