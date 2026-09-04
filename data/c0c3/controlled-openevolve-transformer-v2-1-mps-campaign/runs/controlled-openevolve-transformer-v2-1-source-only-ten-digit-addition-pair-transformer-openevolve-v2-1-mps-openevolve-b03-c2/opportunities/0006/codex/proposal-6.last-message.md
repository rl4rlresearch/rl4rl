MECHANISM: Single-coordinate softmax-invariant key-bias removal

HYPOTHESIS: Fixing only one key-bias coordinate at zero will reduce the model to 1643 parameters while retaining at least 99% accuracy, because any constant key bias is mathematically canceled by attention softmax and this minimal parameterization change should perturb optimization less than removing all eight key-bias coordinates.

INTENDED_EDIT: Preserve the original fused QKV layer and constructor RNG consumption, replace its 24-element bias with 23 learned elements, and reconstruct one fixed-zero key-bias coordinate during the fused projection.

EVIDENCE: The 1644-parameter baseline reached 99.96%, whereas removing all eight theoretically redundant key-bias parameters failed; a one-coordinate reduction is the smallest informative probe of that optimization sensitivity.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep the fused projection and its construction-time RNG consumption,
        # while fixing one softmax-invariant key-bias coordinate at zero.
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