MECHANISM: Absorb the final tied value-bias group into the attention output bias

HYPOTHESIS: Removing the last learned value-bias scalar will reduce the model from 1,603 to 1,602 parameters while retaining at least 99% accuracy after 14,000 steps, because its position-independent contribution can be absorbed by the learned output-projection bias.

INTENDED_EDIT: Store only the eight query-bias parameters, reconstruct the entire value bias as zero, and add 1,000 minimum-learning-rate refinement steps.

EVIDENCE: Removing the preceding tied value-bias group achieved 99.90% at 1,603 parameters after 13,000 steps, while earlier singleton removals also succeeded; the remaining scalar has the same redundant value-bias role.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 15))
=======
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 16))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias.new_zeros(1),
                self.qkv.bias.new_zeros(1),
                self.qkv.bias.new_zeros(1),
=======
                self.qkv.bias.new_zeros(d_model),
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=13000)
=======
    p.add_argument("--train-steps", type=int, default=14000)
>>>>>>> REPLACE