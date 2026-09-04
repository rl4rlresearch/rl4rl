MECHANISM: Cross-path bias tying

HYPOTHESIS: Sharing the final value-bias coordinate with a projection-bias coordinate will reduce the model to 1635 parameters while retaining at least 99% accuracy, because the coordinate remains trainable through both redundant bias paths instead of being fixed at zero.

INTENDED_EDIT: Shorten the query/value bias parameter by one scalar and reconstruct the missing value-bias coordinate from the learned attention output-projection bias.

EVIDENCE: The 1636-parameter model reached 99.98%, but fixing one value-bias coordinate at zero collapsed accuracy to 31.48%; tying that coordinate to another learned, functionally redundant bias tests whether preserving its trainable optimization path avoids the collapse.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model))
        self.qkv.bias = None
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Deduplicate one redundant bias scalar while keeping the missing value
        # coordinate learned through the output-projection bias.
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 1))
        self.qkv.bias = None
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias, v_bias = self.qv_bias.chunk(2)
        fused_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
=======
        q_bias = self.qv_bias[:d_model]
        v_bias = torch.cat((self.qv_bias[d_model:], self.proj.bias[-1:]))
        fused_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
>>>>>>> REPLACE