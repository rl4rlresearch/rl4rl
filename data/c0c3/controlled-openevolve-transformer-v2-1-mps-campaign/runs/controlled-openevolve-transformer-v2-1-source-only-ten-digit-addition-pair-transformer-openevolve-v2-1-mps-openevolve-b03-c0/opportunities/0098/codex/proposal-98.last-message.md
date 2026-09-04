MECHANISM: Query-bias coordinate ablation

HYPOTHESIS: Fixing one query-bias coordinate to zero will reduce the verified model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 steps.

INTENDED_EDIT: Store seven learned query-bias coordinates, reconstruct the eighth as zero, and remove the functionally irrelevant shared key bias without adding gauge-optimizer overhead.

EVIDENCE: The current 1,573-parameter model achieved 99.98% accuracy, while several exact-gauge reductions timed out after adding optimizer or reconstruction complexity; a one-coordinate attention-bias ablation tests the available accuracy margin while preserving the verified training path and runtime profile.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 16))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias.new_zeros(d_model),
            )
        )
=======
        query_bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(1))
        )
        qkv_bias = torch.cat(
            (query_bias, self.qkv.bias.new_zeros(2 * d_model))
        )
>>>>>>> REPLACE