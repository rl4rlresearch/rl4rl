MECHANISM: Shared value/output-bias coordinate

HYPOTHESIS: Tying the retained value-bias coordinate to output-projection bias coordinate 5 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because it preserves the critical learned value-bias pathway while the direct output-bias contribution keeps the shared direction well-conditioned.

INTENDED_EDIT: Remove the separate value-bias parameter, use `proj_bias[5]` for both value and output bias, and optimize the shared attention bias with ordinary AdamW.

EVIDENCE: Removing the remaining value bias collapsed accuracy to 16.83%, whereas the current design achieves 99.93%; sharing that coordinate tests the redundant bias pathway without eliminating the empirically essential value-side signal.

<<<<<<< SEARCH
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 7))
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
=======
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v_bias = torch.cat(
            (
                self.v_bias.new_zeros(5),
                self.v_bias,
                self.v_bias.new_zeros(2),
            )
        )
=======
        v_bias = torch.cat(
            (
                self.proj_bias.new_zeros(5),
                self.proj_bias[5:6],
                self.proj_bias.new_zeros(2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [
        *[block.attn.proj_bias for block in model.blocks],
        *[block.mlp.fc2.bias for block in model.blocks],
    ]
=======
    gauge_parameters = [
        *[block.mlp.fc2.bias for block in model.blocks],
    ]
>>>>>>> REPLACE