MECHANISM: Initialization-preserving elimination of softmax-invariant key bias

HYPOTHESIS: Removing the eight redundant key-bias parameters while preserving the baseline’s constructor RNG consumption and fused QKV computation will retain at least 99% accuracy with 1636 parameters.

INTENDED_EDIT: Keep the full fused QKV weight initialization, replace its 24-element bias with learned query/value biases, and inject a fixed zero key bias during the fused projection.

EVIDENCE: The 1644-parameter baseline reached 99.96%, while the prior key-bias removal reached 98.27%; because key bias cancels under softmax, preserving baseline initialization and fused projection isolates initialization sensitivity as the likely cause.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Construct with the original bias first to preserve the baseline RNG
        # sequence, then retain only the functionally relevant q/v biases.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model))
        self.qkv.bias = None
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        q_bias, v_bias = self.qv_bias.chunk(2)
        fused_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
        qkv = F.linear(x, self.qkv.weight, fused_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE