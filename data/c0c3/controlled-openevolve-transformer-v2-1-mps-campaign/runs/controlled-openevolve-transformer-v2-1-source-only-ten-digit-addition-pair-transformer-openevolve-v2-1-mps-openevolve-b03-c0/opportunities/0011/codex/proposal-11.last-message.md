MECHANISM: Initialization-preserving removal of softmax-invariant key bias

HYPOTHESIS: Eliminating all eight key-bias parameters while preserving the original `nn.Linear` construction RNG trajectory will reduce parameters from 1,620 to 1,612 and retain at least 99% accuracy.

INTENDED_EDIT: Retain the combined QKV layer’s original construction, replace its stored bias with query/value biases only, and reconstruct a zero key bias during the learned attention computation.

EVIDENCE: The current model reaches 99.40% accuracy, while the earlier key-bias removal unexpectedly failed despite key bias being canceled row-wise by softmax. Preserving the original biased-layer construction isolates that mathematical reduction from initialization changes.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Preserve the original constructor's RNG consumption, then retain only
        # the functionally relevant query and value biases.
        self.qkv.bias = nn.Parameter(torch.zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(d_model),
                self.qkv.bias[d_model:],
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE