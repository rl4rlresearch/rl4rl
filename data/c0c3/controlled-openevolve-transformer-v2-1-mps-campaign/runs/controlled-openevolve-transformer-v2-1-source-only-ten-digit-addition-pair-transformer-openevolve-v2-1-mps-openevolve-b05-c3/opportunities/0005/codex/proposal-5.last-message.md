MECHANISM: Initialization-preserving removal of softmax-invariant key bias

HYPOTHESIS: Removing all eight key-bias parameters while preserving the baseline constructor RNG stream will produce 1636 parameters and retain at least 99% accuracy, because zero key bias is functionally invisible to causal softmax and all remaining weights receive the proven baseline initialization.

INTENDED_EDIT: Keep the fused QKV weight and baseline initialization sequence, replace its 24-element bias with 16 learned query/value biases, and insert a fixed zero key bias only during the forward pass.

EVIDENCE: The 1644-parameter baseline reached 99.96%, while the earlier 1636-parameter key-bias removal reached 86.15%; since key bias mathematically adds only a row-wise softmax constant, the failed result motivates preserving the baseline RNG stream and initialization exactly while removing the same redundant degrees of freedom.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Construct with the baseline shape first so subsequent modules retain
        # exactly the proven initialization RNG stream. Only query and value
        # biases are learned; key bias is softmax-invariant.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        q_bias, v_bias = self.qkv.bias.chunk(2, dim=0)
        bias = torch.cat((q_bias, self.qkv.bias.new_zeros(d_model), v_bias))
        qkv = F.linear(x, self.qkv.weight, bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE