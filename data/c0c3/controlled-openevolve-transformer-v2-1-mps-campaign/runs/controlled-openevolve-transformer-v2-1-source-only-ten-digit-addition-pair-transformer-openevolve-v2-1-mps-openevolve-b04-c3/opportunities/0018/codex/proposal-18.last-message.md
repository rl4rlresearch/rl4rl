MECHANISM: Complete softmax-invariant key-bias elimination

HYPOTHESIS: Removing all eight redundant key-projection bias coordinates and training for 45,000 steps will retain at least 99% accuracy with 1,588 learned parameters.

INTENDED_EDIT: Reconstruct the entire key-bias vector as fixed zeros while preserving constructor RNG consumption, and increase default training from 30,000 to 45,000 steps.

EVIDENCE: The seven-coordinate design achieved 99.96% accuracy with 1,589 parameters at 39,999 steps; completing the exact key-bias quotient with 5,000 additional steps is the closest supported extension.

<<<<<<< SEARCH
        # Five key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 5))
=======
        # All key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(5),
                self.qkv.bias[d_model:],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(d_model),
                self.qkv.bias[d_model:],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=30000)
=======
    p.add_argument("--train-steps", type=int, default=45000)
>>>>>>> REPLACE