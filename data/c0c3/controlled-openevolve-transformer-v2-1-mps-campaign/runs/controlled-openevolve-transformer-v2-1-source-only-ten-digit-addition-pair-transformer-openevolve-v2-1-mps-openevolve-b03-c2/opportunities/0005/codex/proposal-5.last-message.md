MECHANISM: RNG-preserving fused key-bias elimination

HYPOTHESIS: Removing only the softmax-invariant key bias while preserving the fused QKV operation and seeded initialization trajectory will achieve at least 99% accuracy with 1636 parameters.

INTENDED_EDIT: Instantiate the original biased fused projection to preserve RNG consumption, discard its bias, add learned query/value biases, and reconstruct the fused bias with a fixed zero key segment.

EVIDENCE: The 1644-parameter model achieved 99.96%, while the prior 1636-parameter key-bias trial reached only 94.61% despite the key bias being functionally redundant; preserving the original fused projection and initialization trajectory isolates that redundancy more precisely.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Construct with bias first to preserve the original seeded RNG
        # consumption, then retain only the useful query/value coordinates.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qkv_bias = torch.cat(
            (
                self.qv_bias[:d_model],
                self.qv_bias.new_zeros((d_model,)),
                self.qv_bias[d_model:],
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE