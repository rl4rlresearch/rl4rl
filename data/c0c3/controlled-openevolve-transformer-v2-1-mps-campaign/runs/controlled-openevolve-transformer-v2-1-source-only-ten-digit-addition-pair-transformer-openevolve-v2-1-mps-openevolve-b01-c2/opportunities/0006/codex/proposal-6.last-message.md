MECHANISM: Fused zero-key-bias reparameterization

HYPOTHESIS: Removing the eight softmax-invariant key-bias parameters while retaining the baseline’s constructor RNG stream and single fused linear bias addition will achieve at least 99% accuracy in 5,000 steps with 1,636 parameters.

INTENDED_EDIT: Preserve fused QKV construction, replace its bias with learned query/value components, synthesize the zero key component, and pass the complete bias through one fused `F.linear` call.

EVIDENCE: The 1,644-parameter baseline achieved 99.96%, while the prior 1,636-parameter implementation reached 97.79% after applying biases separately; preserving the baseline fused bias-add computation targets that remaining numerical difference.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Construct the baseline projection first to preserve its RNG stream, then
        # retain only the effective query and value bias parameters.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qv_bias = self.qkv.bias
        fused_bias = torch.cat(
            (qv_bias[:d_model], qv_bias.new_zeros(d_model), qv_bias[d_model:])
        )
        qkv = F.linear(x, self.qkv.weight, fused_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE