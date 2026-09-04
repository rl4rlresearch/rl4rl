MECHANISM: Incremental absorption of a singleton value bias

HYPOTHESIS: Removing the remaining singleton value-bias scalar will reduce the model from 1,605 to 1,604 parameters while retaining at least 99% accuracy after 12,000 steps, because its position-independent attention contribution can be absorbed by the learned output-projection bias.

INTENDED_EDIT: Store 10 QKV bias parameters, reconstruct value coordinate 6 as zero while preserving all other bias mappings, and add 1,000 low-learning-rate refinement steps.

EVIDENCE: Removing the previous singleton value-bias coordinate achieved 99.97% at 1,605 parameters after 11,000 steps, directly validating the same redundancy and incremental training strategy.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 13))
=======
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 14))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias[d_model : d_model + 2],
                self.qkv.bias[d_model + 1 :],
                self.qkv.bias.new_zeros(1),
=======
                self.qkv.bias[d_model : d_model + 2],
                self.qkv.bias[d_model + 1 : d_model + 2],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias.new_zeros(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=11000)
=======
    p.add_argument("--train-steps", type=int, default=12000)
>>>>>>> REPLACE