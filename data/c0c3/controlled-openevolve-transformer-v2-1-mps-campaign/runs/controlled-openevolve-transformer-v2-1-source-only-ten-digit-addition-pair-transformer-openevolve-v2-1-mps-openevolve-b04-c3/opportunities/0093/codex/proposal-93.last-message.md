MECHANISM: Output-side shared-value basis gauge fixing

HYPOTHESIS: Fixing one trailing attention-output projection coefficient at zero will preserve at least 99% accuracy after 45,000 updates while reducing the verified 1,141-parameter model to 1,140 learned parameters.

INTENDED_EDIT: Learn 63 of the 64 attention-output projection weights, reconstruct the final coefficient as zero, and preserve full-matrix initialization RNG consumption.

EVIDENCE: One shared-value coefficient was removed successfully at 1,141 parameters, demonstrating redundancy in the value/output basis; because removing a second adjacent value coefficient failed, the complementary learned output projection is the most informative location for the next one-parameter constraint.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        # The final two attention-output bias coordinates are fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 2))
=======
        self.proj = nn.Linear(d_model, d_model)
        # Fix one coefficient on the output side of the shared-value basis.
        # Constructing the full Linear preserves constructor RNG consumption.
        self.proj.weight = nn.Parameter(torch.empty(d_model * d_model - 1))
        self.proj.fixed_weight_trim = (0, 1)
        # The final two attention-output bias coordinates are fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_bias = F.pad(self.proj.bias, (0, 2))
        y = F.linear(y, self.proj.weight, proj_bias)
=======
        proj_weight = F.pad(self.proj.weight, (0, 1)).view(d_model, d_model)
        proj_bias = F.pad(self.proj.bias, (0, 2))
        y = F.linear(y, proj_weight, proj_bias)
>>>>>>> REPLACE