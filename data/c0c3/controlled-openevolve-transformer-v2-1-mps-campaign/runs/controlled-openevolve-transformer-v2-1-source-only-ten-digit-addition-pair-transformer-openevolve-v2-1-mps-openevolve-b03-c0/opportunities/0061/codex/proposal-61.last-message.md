MECHANISM: Fixed-seed trajectory-preserving second MLP output-column gauge

HYPOTHESIS: Gauge-anchoring a second `fc2` column while exactly preserving the successful 1,595-parameter design’s initialization draw stream will retain at least 99% accuracy at 1,594 parameters after 21,000 steps.

INTENDED_EDIT: Initialize `fc2` in the current storage geometry, split its final two columns into gauge coordinates only afterward, and train both with `GaugeAdamW`.

EVIDENCE: The first `fc2` anchor achieved 100% accuracy at 1,595 parameters, whereas the earlier second anchor reached 74.07% after perturbing the fixed-seed parameter layout; this patch isolates the same exact gauge without changing the successful initialization trajectory.

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.weight = nn.Parameter(torch.empty(d_model, d_ff - 1))
        self.fc2_col = nn.Parameter(torch.empty(d_model - 1))
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2._defer_two_column_gauge = True
        self.fc2.register_parameter("unused", None)
        self.register_parameter("fc2_col_2", None)
        self.register_parameter("fc2_col", None)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        fc2_col = torch.cat(
            (self.fc2_col, self.fc2_col.new_zeros(1))
        ).unsqueeze(1)
        fc2_weight = torch.cat((self.fc2.weight, fc2_col), dim=1)
=======
        fc2_col_2 = torch.cat(
            (self.fc2_col_2, self.fc2_col_2.new_zeros(1))
        ).unsqueeze(1)
        fc2_col = torch.cat(
            (self.fc2_col, self.fc2_col.new_zeros(1))
        ).unsqueeze(1)
        fc2_weight = torch.cat(
            (self.fc2.weight, fc2_col_2, fc2_col), dim=1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_fc2_col = block.mlp.fc2_col.new_empty(cfg.d_model)
                nn.init.normal_(full_fc2_col, mean=0.0, std=0.02)
                full_fc2_col.sub_(full_fc2_col[-1].clone())
                block.mlp.fc2_col.copy_(full_fc2_col[:-1])
=======
                full_fc2_col = block.mlp.fc2.weight.new_empty(cfg.d_model)
                nn.init.normal_(full_fc2_col, mean=0.0, std=0.02)
                full_fc2_col_2 = block.mlp.fc2.weight[:, -2].detach().clone()
                base_fc2_weight = block.mlp.fc2.weight[:, :-2].detach().clone()
                block.mlp.fc2.weight = nn.Parameter(base_fc2_weight)

                full_fc2_col_2.sub_(full_fc2_col_2[-1].clone())
                block.mlp.fc2_col_2 = nn.Parameter(
                    full_fc2_col_2[:-1].clone()
                )
                full_fc2_col.sub_(full_fc2_col[-1].clone())
                block.mlp.fc2_col = nn.Parameter(full_fc2_col[:-1].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if getattr(module, "_defer_two_column_gauge", False):
                initialized_weight = module.weight.new_empty(
                    module.out_features, module.in_features - 1
                )
                nn.init.normal_(initialized_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight[:, :-1].copy_(initialized_weight)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    ] + [
        block.mlp.fc2_col for block in model.blocks
    ]
=======
    ] + [
        block.mlp.fc2_col_2 for block in model.blocks
    ] + [
        block.mlp.fc2_col for block in model.blocks
    ]
>>>>>>> REPLACE