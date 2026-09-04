MECHANISM: Orthonormal LayerNorm-null attention projection gauge

HYPOTHESIS: Reparameterizing the first attention output-projection column in an orthonormal zero-sum basis will reduce the model to 1619 parameters while achieving at least 99% accuracy, because it removes the same exact LayerNorm-null direction as the narrowly failing 98.92% trial without the anisotropic optimization geometry of anchored contrasts.

INTENDED_EDIT: Replace the first attention projection column with seven learned Helmert-basis coordinates, retain the other seven columns unchanged, and reconstruct the centered original initialization without altering the RNG sequence.

EVIDENCE: The attention projection-bias gauge reached 99.89% at 1620 parameters, proving this residual common-offset symmetry is usable; the first projection-column attempt reached 98.92%, so improving that exact gauge’s conditioning is more strongly motivated than testing the second column again, which reached only 72.45%.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj.bias = None
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model)
        self.proj_first_column = nn.Parameter(torch.empty(d_model - 1))
        self.proj_rest = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.proj.weight = None
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj.bias = None

        # Orthonormal Helmert basis for the featurewise zero-sum subspace.
        proj_basis = torch.zeros(d_model, d_model - 1)
        for column in range(d_model - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            proj_basis[: column + 1, column] = 1.0 / scale
            proj_basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("proj_basis", proj_basis, persistent=False)

        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_bias = torch.cat((self.proj_bias, self.proj_bias.new_zeros(1)))
        proj_bias = proj_bias - proj_bias.mean()
        y = F.linear(y, self.proj.weight, proj_bias)
=======
        proj_bias = torch.cat((self.proj_bias, self.proj_bias.new_zeros(1)))
        proj_bias = proj_bias - proj_bias.mean()
        first_column = self.proj_basis @ self.proj_first_column
        proj_weight = torch.cat(
            (first_column.unsqueeze(1), self.proj_rest),
            dim=1,
        )
        y = F.linear(y, proj_weight, proj_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, CenteredPositionalEmbedding):
            full_weight = module.fourth_column.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, MLP):
=======
        elif isinstance(module, CenteredPositionalEmbedding):
            full_weight = module.fourth_column.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, CausalSelfAttention):
            full_weight = module.proj_first_column.new_empty(
                module.proj.out_features, module.proj.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            first_column = full_weight[:, 0] - full_weight[:, 0].mean()
            with torch.no_grad():
                module.proj_first_column.copy_(
                    module.proj_basis.transpose(0, 1) @ first_column
                )
                module.proj_rest.copy_(full_weight[:, 1:])
        elif isinstance(module, MLP):
>>>>>>> REPLACE