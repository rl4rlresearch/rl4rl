MECHANISM: Third query-bias coordinate ablation

HYPOTHESIS: Learning five query-bias coordinates while retaining the qualified gauge-aware positional parameterization will produce 1602 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Fix one additional trailing query-bias coordinate at zero, leaving all weight tensors, positional quotient updates, initialization streams, and training behavior unchanged.

EVIDENCE: The analogous reduction from seven to six query-bias coordinates improved the qualified gauge-aware design from 99.8% at 1604 parameters to 99.94% at 1603 parameters, making another one-scalar localized ablation the most informative next test.

<<<<<<< SEARCH
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Six query coordinates are
        # learned; the remaining query coordinates and key/value biases are zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 2))
=======
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Five query coordinates are
        # learned; the remaining query coordinates and key/value biases are zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 2))
        )
=======
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 3))
        )
>>>>>>> REPLACE