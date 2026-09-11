MECHANISM: Sixth-coordinate softmax-invariant key-bias quotient

HYPOTHESIS: Removing a sixth redundant key-projection bias coordinate and training for 35,000 steps will retain at least 99% accuracy with 1,590 learned parameters.

INTENDED_EDIT: Reconstruct six fixed-zero key-bias coordinates in the fused QKV projection while preserving constructor RNG consumption, and increase default training from 10,000 to 35,000 steps.

EVIDENCE: The five-coordinate design achieved 99.85% accuracy with 1,591 parameters at 29,999 steps; the successful progression from two through five removed coordinates supports testing one additional coordinate with 5,000 more steps.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Six key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 6))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(6),
                self.qkv.bias[d_model:],
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=10000)
=======
    p.add_argument("--train-steps", type=int, default=35000)
>>>>>>> REPLACE