MECHANISM: Initialization-preserving removal of softmax-invariant key bias

HYPOTHESIS: Keeping the fused QKV weight layout and original initialization stream while removing only the 8 ineffective key-bias parameters will retain at least 99% accuracy with 1,636 parameters.

INTENDED_EDIT: Resize the fused projection’s learned bias to query and value components only, then apply those components explicitly around the unchanged fused weight projection.

EVIDENCE: The 1,644-parameter baseline reached 99.96%, while capacity reductions failed; the prior key-bias removal reached 86.2%, motivating a more tightly controlled implementation that preserves the successful model’s fused projection layout and initialization.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Retain the fused weight and its initialization stream, but learn only
        # query and value biases; a key bias is softmax-invariant.
        self.qkv.bias = nn.Parameter(torch.zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qkv = F.linear(x, self.qkv.weight)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias, v_bias = self.qkv.bias.chunk(2, dim=-1)
        q = q + q_bias
        v = v + v_bias
>>>>>>> REPLACE