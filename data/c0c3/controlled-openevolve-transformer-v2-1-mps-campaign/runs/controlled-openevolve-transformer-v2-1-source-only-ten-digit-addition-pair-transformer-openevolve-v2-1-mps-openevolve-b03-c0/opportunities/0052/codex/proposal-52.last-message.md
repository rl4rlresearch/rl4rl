MECHANISM: Incremental initialization-preserving attention-output column gauge

HYPOTHESIS: Anchoring a second attention-projection input column with equivalent full eight-coordinate initialization and implicit AdamW updates will reduce the model from 1,600 to 1,599 parameters while retaining at least 99% accuracy after 17,000 steps.

INTENDED_EDIT: Split another projection column into seven learned relative coordinates plus a zero anchor, initialize both anchored columns from full i.i.d. columns, optimize each with GaugeAdamW, and add 1,000 low-rate refinement steps.

EVIDENCE: Initialization-preserving anchoring rescued the first projection-column gauge from 40.05% to 99.98% accuracy at 1,600 parameters, directly validating the same exact symmetry and optimizer treatment used here.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.proj.weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.proj_col = nn.Parameter(torch.empty(d_model - 1))
=======
        self.proj = nn.Linear(d_model, d_model)
        self.proj.weight = nn.Parameter(torch.empty(d_model, d_model - 2))
        self.proj_col_2 = nn.Parameter(torch.empty(d_model - 1))
        self.proj_col = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_col = torch.cat(
            (self.proj_col, self.proj_col.new_zeros(1))
        ).unsqueeze(1)
        proj_weight = torch.cat((self.proj.weight, proj_col), dim=1)
=======
        proj_col_2 = torch.cat(
            (self.proj_col_2, self.proj_col_2.new_zeros(1))
        ).unsqueeze(1)
        proj_col = torch.cat(
            (self.proj_col, self.proj_col.new_zeros(1))
        ).unsqueeze(1)
        proj_weight = torch.cat(
            (self.proj.weight, proj_col_2, proj_col), dim=1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            for block in self.blocks:
                full_proj_col = block.attn.proj_col.new_empty(cfg.d_model)
                nn.init.normal_(full_proj_col, mean=0.0, std=0.02)
                full_proj_col.sub_(full_proj_col[-1].clone())
                block.attn.proj_col.copy_(full_proj_col[:-1])
=======
            for block in self.blocks:
                for proj_col in (block.attn.proj_col_2, block.attn.proj_col):
                    full_proj_col = proj_col.new_empty(cfg.d_model)
                    nn.init.normal_(full_proj_col, mean=0.0, std=0.02)
                    full_proj_col.sub_(full_proj_col[-1].clone())
                    proj_col.copy_(full_proj_col[:-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
    ] + [
        block.attn.proj_col for block in model.blocks
    ]
=======
    ] + [
        block.attn.proj_col_2 for block in model.blocks
    ] + [
        block.attn.proj_col for block in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=16000)
=======
    p.add_argument("--train-steps", type=int, default=17000)
>>>>>>> REPLACE