MECHANISM: Absorb a redundant value-bias coordinate into the attention output bias

HYPOTHESIS: Removing the final singleton value-bias coordinate will reduce the model from 1,606 to 1,605 parameters while retaining at least 99% accuracy after 11,000 steps, because a position-independent value bias contributes only a constant after attention and can be represented by the learned output-projection bias.

INTENDED_EDIT: Store 11 QKV bias parameters, reconstruct the final value-bias coordinate as zero while preserving every other validated bias mapping, and add 1,000 low-learning-rate refinement steps.

EVIDENCE: The 1,606-parameter design reached 99.8%, while anchoring a second `fc1` neuron fell to 88.83%; the successful gauge-aware attention-output bias at 1,607 parameters motivates testing this distinct exact redundancy, since that bias can absorb the removed value offset.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 12))
=======
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 13))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias[d_model : d_model + 2],
                self.qkv.bias[d_model + 1 :],
=======
                self.qkv.bias[d_model : d_model + 2],
                self.qkv.bias[d_model + 1 :],
                self.qkv.bias.new_zeros(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=10000)
=======
    p.add_argument("--train-steps", type=int, default=11000)
>>>>>>> REPLACE