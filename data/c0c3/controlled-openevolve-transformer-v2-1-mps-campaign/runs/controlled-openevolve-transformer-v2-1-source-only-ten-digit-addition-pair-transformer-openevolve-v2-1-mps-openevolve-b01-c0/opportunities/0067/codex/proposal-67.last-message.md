MECHANISM: First-column attention-output common-mode gauge removal

HYPOTHESIS: Centering the first attention projection column will reduce the model to 1605 parameters while maintaining at least 99% accuracy, because its removed all-ones component propagates only as a featurewise common-mode shift that is eliminated by downstream LayerNorms.

INTENDED_EDIT: Represent the first attention output-projection column with seven learned coordinates and reconstruct its eighth coordinate to enforce zero column sum, while preserving the original initialization draw.

EVIDENCE: The verified 1606-parameter design achieves 99.89% accuracy with the same centered-column reconstruction applied successfully to the first two `fc2` columns; testing the first column of the attention projection transfers that proven gauge reduction to the other residual projection.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj.bias = None
=======
        self.proj = nn.Linear(d_model, d_model)
        self.proj_first_column = nn.Parameter(torch.empty(d_model - 1))
        self.proj_rest = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.proj.weight = None
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj.bias = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_bias = torch.cat((self.proj_bias, self.proj_bias.new_zeros(1)))
        proj_bias = proj_bias - proj_bias.mean()
        y = F.linear(y, self.proj.weight, proj_bias)
=======
        first_final_coordinate = -self.proj_first_column.sum().reshape(1)
        first_column = torch.cat(
            (self.proj_first_column, first_final_coordinate)
        )
        proj_weight = torch.cat(
            (first_column.unsqueeze(1), self.proj_rest),
            dim=1,
        )
        proj_bias = torch.cat((self.proj_bias, self.proj_bias.new_zeros(1)))
        proj_bias = proj_bias - proj_bias.mean()
        y = F.linear(y, proj_weight, proj_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MLP):
            full_weight = module.fc2_first_column.new_empty(
                module.fc2.out_features, module.fc2.in_features
            )
=======
        elif isinstance(module, CausalSelfAttention):
            full_weight = module.proj_first_column.new_empty(
                module.proj.out_features, module.proj.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            first_column = full_weight[:, 0] - full_weight[:, 0].mean()
            with torch.no_grad():
                module.proj_first_column.copy_(first_column[:-1])
                module.proj_rest.copy_(full_weight[:, 1:])
        elif isinstance(module, MLP):
            full_weight = module.fc2_first_column.new_empty(
                module.fc2.out_features, module.fc2.in_features
            )
>>>>>>> REPLACE