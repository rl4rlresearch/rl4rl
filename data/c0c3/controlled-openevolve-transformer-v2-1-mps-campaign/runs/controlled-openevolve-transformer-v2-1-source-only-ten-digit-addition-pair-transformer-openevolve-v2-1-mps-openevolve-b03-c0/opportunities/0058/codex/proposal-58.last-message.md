MECHANISM: Initialization-preserving MLP output-column gauge

HYPOTHESIS: Anchoring one `fc2` input column’s uniform-output coordinate will reduce the model from 1,596 to 1,595 parameters while retaining at least 99% accuracy after 21,000 steps, because adding a uniform vector to that column only creates positionwise residual shifts removed by subsequent LayerNorm.

INTENDED_EDIT: Store the final `fc2` column as seven relative coordinates plus a zero anchor, initialize it from an equivalent full eight-coordinate column, optimize it with GaugeAdamW, and add 1,000 low-rate refinement steps.

EVIDENCE: Initialization-preserving attention-output column gauges achieved 99.98% and 99.96%, while the current gauge-anchored MLP output bias achieved 99.96%; together these validate the same uniform-output symmetry and optimizer treatment in the MLP without repeating the failed third attention-column or positional-row anchor.

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)
=======
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.weight = nn.Parameter(torch.empty(d_model, d_ff - 1))
        self.fc2_col = nn.Parameter(torch.empty(d_model - 1))
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        fc2_bias = torch.cat((self.fc2.bias, self.fc2.bias.new_zeros(1)))
        output = F.linear(F.gelu(hidden), self.fc2.weight, fc2_bias)
=======
        fc2_col = torch.cat(
            (self.fc2_col, self.fc2_col.new_zeros(1))
        ).unsqueeze(1)
        fc2_weight = torch.cat((self.fc2.weight, fc2_col), dim=1)
        fc2_bias = torch.cat((self.fc2.bias, self.fc2.bias.new_zeros(1)))
        output = F.linear(F.gelu(hidden), fc2_weight, fc2_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
            for block in self.blocks:
                for proj_col in (block.attn.proj_col_2, block.attn.proj_col):
                    full_proj_col = proj_col.new_empty(cfg.d_model)
                    nn.init.normal_(full_proj_col, mean=0.0, std=0.02)
                    full_proj_col.sub_(full_proj_col[-1].clone())
                    proj_col.copy_(full_proj_col[:-1])

            full_token_weight = self.token_emb.weight.new_empty(
=======
            for block in self.blocks:
                for proj_col in (block.attn.proj_col_2, block.attn.proj_col):
                    full_proj_col = proj_col.new_empty(cfg.d_model)
                    nn.init.normal_(full_proj_col, mean=0.0, std=0.02)
                    full_proj_col.sub_(full_proj_col[-1].clone())
                    proj_col.copy_(full_proj_col[:-1])

                full_fc2_col = block.mlp.fc2_col.new_empty(cfg.d_model)
                nn.init.normal_(full_fc2_col, mean=0.0, std=0.02)
                full_fc2_col.sub_(full_fc2_col[-1].clone())
                block.mlp.fc2_col.copy_(full_fc2_col[:-1])

            full_token_weight = self.token_emb.weight.new_empty(
>>>>>>> REPLACE

<<<<<<< SEARCH
    ] + [
        block.attn.proj_col for block in model.blocks
    ]
=======
    ] + [
        block.attn.proj_col for block in model.blocks
    ] + [
        block.mlp.fc2_col for block in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=20000)
=======
    p.add_argument("--train-steps", type=int, default=21000)
>>>>>>> REPLACE