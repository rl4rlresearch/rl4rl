MECHANISM: Gauge-fixed unit query offset

HYPOTHESIS: Fixing each head’s scalar query offset to one will reduce the model from 1,378 to 1,376 parameters while retaining at least 99% accuracy at 52,000 steps, because inverse query/key rescaling can absorb any nonzero per-head scalar without reducing attention expressivity.

INTENDED_EDIT: Remove the two learned per-head query-bias parameters and replace their broadcast with a fixed unit offset, preserving all optimized operators and training settings.

EVIDENCE: The current gauge-fixed scalar-query model achieved 100% accuracy with 1,378 parameters; unlike earlier 1,376-parameter attempts that changed LayerNorm or projection kernels and timed out, this removes only a query/key scaling redundancy from the verified implementation.

<<<<<<< SEARCH
        self.qkv = MeanZeroInputLinear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(n_head))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = MeanZeroInputLinear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias.view(1, self.n_head, 1, 1)
=======
        q = q + 1.0
>>>>>>> REPLACE