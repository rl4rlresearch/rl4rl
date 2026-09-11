MECHANISM: Longer convergence for single-coordinate key-bias elimination

HYPOTHESIS: The 1,643-parameter model will exceed 99% accuracy when trained for 7,500 steps, because its 96.07% result at 4,999 steps indicates substantially preserved capacity but incomplete convergence.

INTENDED_EDIT: Remove one mathematically redundant key-projection bias coordinate while preserving constructor RNG consumption and fused QKV evaluation, then increase the default training duration from 5,000 to 7,500 steps.

EVIDENCE: The single-coordinate key-bias reduction reached 96.07%, substantially outperforming all other reduced designs; extending optimization directly tests whether its remaining gap is convergence-related rather than a capacity limit.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # The first key-bias coordinate is softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model:],
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=5000)
=======
    p.add_argument("--train-steps", type=int, default=7500)
>>>>>>> REPLACE