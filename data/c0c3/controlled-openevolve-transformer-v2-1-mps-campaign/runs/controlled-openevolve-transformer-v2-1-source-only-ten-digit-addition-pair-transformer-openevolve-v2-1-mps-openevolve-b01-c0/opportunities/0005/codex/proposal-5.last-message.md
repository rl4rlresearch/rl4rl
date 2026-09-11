MECHANISM: Granular value-bias ablation

HYPOTHESIS: Removing only one of the eight value-bias parameters will reduce the model to 1635 parameters while retaining at least 99% accuracy, because the 1636-parameter design has 99.98% accuracy and preserves seven of the eight optimization-relevant value offsets whose complete removal caused collapse.

INTENDED_EDIT: Shorten the combined query/value bias by one scalar and append a fixed zero to reconstruct the final value-bias coordinate during fused QKV projection.

EVIDENCE: Removing all eight value-bias parameters reduced accuracy to 41.07%, while retaining them produced 99.98%; a one-coordinate ablation is the smallest informative test of how much of that redundant parameterization training requires.

<<<<<<< SEARCH
        # Construct with the original bias first to preserve the baseline RNG
        # sequence, then retain only the functionally relevant q/v biases.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model))
        self.qkv.bias = None
=======
        # Construct with the original bias first to preserve the baseline RNG
        # sequence, then retain the query bias and all but one value coordinate.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 1))
        self.qkv.bias = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias, v_bias = self.qv_bias.chunk(2)
        fused_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
=======
        q_bias = self.qv_bias[:d_model]
        v_bias = F.pad(self.qv_bias[d_model:], (0, 1))
        fused_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
>>>>>>> REPLACE