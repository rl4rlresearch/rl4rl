MECHANISM: Seventh-coordinate softmax-invariant key-bias quotient

HYPOTHESIS: Removing a seventh redundant key-projection bias coordinate and training for 40,000 steps will retain at least 99% accuracy with 1,589 learned parameters.

INTENDED_EDIT: Reconstruct seven fixed-zero key-bias coordinates in the fused QKV projection while preserving constructor RNG consumption, and increase default training from 35,000 to 40,000 steps.

EVIDENCE: The six-coordinate design achieved 100% accuracy with 1,590 parameters at 34,999 steps; the successful progression from two through six removed coordinates supports one further coordinate reduction paired with 5,000 additional training steps.

<<<<<<< SEARCH
        # Six key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 6))
=======
        # Seven key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias.new_zeros(6),
=======
                self.qkv.bias.new_zeros(7),
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=35000)
=======
    p.add_argument("--train-steps", type=int, default=40000)
>>>>>>> REPLACE