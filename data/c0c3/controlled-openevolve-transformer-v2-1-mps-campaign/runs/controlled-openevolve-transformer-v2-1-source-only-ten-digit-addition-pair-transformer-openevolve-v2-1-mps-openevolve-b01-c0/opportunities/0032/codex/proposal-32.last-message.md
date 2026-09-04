MECHANISM: LayerNorm-null attention projection-column gauge

HYPOTHESIS: Centering one attention output-projection weight column will reduce the model from 1620 to 1619 parameters while retaining at least 99% accuracy, because its removed output-coordinate mean contributes only a tokenwise common residual offset that is invisible to subsequent LayerNorms.

INTENDED_EDIT: Represent the first attention projection column with seven learned zero-sum contrasts, retain the other columns unchanged, and reconstruct its original centered initialization without changing the RNG sequence.

EVIDENCE: Centering the attention projection bias achieved 99.89% at 1620 parameters through the same residual-stream gauge; changing only one projection column is conservative given that two analogous `fc2` column gauges passed while a third collapsed.

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
            (first_column.unsqueeze(1), self.proj_rest), dim=1
        )
        proj_bias = torch.cat((self.proj_bias, self.proj_bias.new_zeros(1)))
        proj_bias = proj_bias - proj_bias.mean()
        y = F.linear(y, proj_weight, proj_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MLP):
            full_weight = module.fc2_first_column.new_empty(
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
>>>>>>> REPLACE