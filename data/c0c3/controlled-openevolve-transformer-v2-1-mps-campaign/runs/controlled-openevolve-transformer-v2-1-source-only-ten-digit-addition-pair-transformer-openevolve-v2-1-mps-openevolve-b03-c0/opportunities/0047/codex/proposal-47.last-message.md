MECHANISM: Incremental absorption of a tied value-bias group

HYPOTHESIS: Removing the value-bias scalar shared by coordinates 4 and 5 will reduce the model from 1,604 to 1,603 parameters while retaining at least 99% accuracy after 13,000 steps, because its position-independent attention contribution can be absorbed by the learned output-projection bias.

INTENDED_EDIT: Store nine QKV bias parameters, reconstruct value coordinates 4 and 5 as zero while preserving the remaining validated mappings, and add 1,000 low-learning-rate refinement steps.

EVIDENCE: Successive removal of two singleton value-bias parameters produced 99.97% at 1,605 parameters and 99.35% at 1,604 parameters; this directly supports testing the next incremental value-bias absorption rather than repeating the failed second-neuron anchoring.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 14))
=======
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 15))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 2],
                self.qkv.bias[d_model + 1 : d_model + 2],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias.new_zeros(1),
=======
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias.new_zeros(1),
                self.qkv.bias.new_zeros(1),
                self.qkv.bias.new_zeros(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=12000)
=======
    p.add_argument("--train-steps", type=int, default=13000)
>>>>>>> REPLACE