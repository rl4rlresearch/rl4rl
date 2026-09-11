MECHANISM: Incremental fourth-coordinate key-bias quotient

HYPOTHESIS: Removing a fourth softmax-invariant key-bias coordinate and training for 25,000 steps will retain at least 99% accuracy with 1,592 learned parameters.

INTENDED_EDIT: Reconstruct four fixed-zero key-bias coordinates in the fused QKV projection and extend default training from 20,000 to 25,000 steps.

EVIDENCE: The three-coordinate design achieved 99.96% accuracy with 1,593 parameters at 19,999 steps, continuing the successful progression from two removed coordinates at 14,999 steps; another single-coordinate reduction with 5,000 additional steps is the closest supported extension.

<<<<<<< SEARCH
        # Three key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 3))
=======
        # Four key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias.new_zeros(3),
=======
                self.qkv.bias.new_zeros(4),
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=20000)
=======
    p.add_argument("--train-steps", type=int, default=25000)
>>>>>>> REPLACE