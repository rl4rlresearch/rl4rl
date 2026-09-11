MECHANISM: Eliminate value/output bias redundancy

HYPOTHESIS: Removing the eight value-bias parameters will reduce the model from 1636 to 1628 parameters while retaining at least 99% accuracy, because attention-weighted value bias is a position-independent offset already representable by the output projection bias.

INTENDED_EDIT: Preserve constructor RNG consumption and fused QKV computation, but retain only the learned query bias and inject fixed zero key/value biases.

EVIDENCE: Initialization-preserving key-bias elimination achieved 99.98% accuracy at 1636 parameters, while narrowing the MLP failed; this motivates removing another analytically redundant attention bias without reducing learned MLP capacity.

<<<<<<< SEARCH
        # Construct with the original bias first to preserve the baseline RNG
        # sequence, then retain only the functionally relevant q/v biases.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model))
        self.qkv.bias = None
=======
        # Construct with the original bias first to preserve the baseline RNG
        # sequence, then retain only the functionally independent query bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.qkv.bias = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias, v_bias = self.qv_bias.chunk(2)
        fused_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
=======
        zero_bias = torch.zeros_like(self.q_bias)
        fused_bias = torch.cat((self.q_bias, zero_bias, zero_bias))
>>>>>>> REPLACE