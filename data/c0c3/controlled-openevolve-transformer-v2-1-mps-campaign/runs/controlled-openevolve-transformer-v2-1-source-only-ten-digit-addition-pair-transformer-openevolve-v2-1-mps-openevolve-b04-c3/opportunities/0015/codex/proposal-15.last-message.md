MECHANISM: Fifth-coordinate softmax-invariant key-bias quotient

HYPOTHESIS: Removing a fifth redundant key-projection bias coordinate and training for 30,000 steps will retain at least 99% accuracy with 1,591 learned parameters.

INTENDED_EDIT: Reconstruct five fixed-zero key-bias coordinates in the fused QKV projection and increase default training from 15,000 to 30,000 steps.

EVIDENCE: The four-coordinate design achieved 99.95% accuracy with 1,592 parameters at 24,999 steps, extending the successful progression in which each additional removed coordinate was paired with 5,000 more training steps.

<<<<<<< SEARCH
        # Two key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 2))
=======
        # Five key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(2),
                self.qkv.bias[d_model:],
=======
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(5),
                self.qkv.bias[d_model:],
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=15000)
=======
    p.add_argument("--train-steps", type=int, default=30000)
>>>>>>> REPLACE