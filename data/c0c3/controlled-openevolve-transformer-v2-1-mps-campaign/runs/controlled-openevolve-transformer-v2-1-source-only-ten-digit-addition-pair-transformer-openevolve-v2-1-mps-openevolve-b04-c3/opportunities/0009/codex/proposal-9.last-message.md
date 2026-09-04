MECHANISM: Two-coordinate softmax-invariant key-bias elimination

HYPOTHESIS: Removing two redundant key-projection bias coordinates and training for 10,000 steps will achieve at least 99% accuracy with 1,642 parameters.

INTENDED_EDIT: Preserve the original constructor RNG stream, reconstruct two fixed-zero key-bias coordinates in the fused QKV operation, and increase default training from 5,000 to 10,000 steps.

EVIDENCE: The one-coordinate reduction reached 96.07% at 4,999 steps and 99.98% at 7,499 steps with 1,643 parameters, showing that incremental key-bias elimination preserves capacity but needs longer optimization.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Two key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 2))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        bsz, seqlen, d_model = x.shape
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(2),
                self.qkv.bias[d_model:],
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=5000)
=======
    p.add_argument("--train-steps", type=int, default=10000)
>>>>>>> REPLACE