MECHANISM: Incremental initialization-preserving attention-output column gauge

HYPOTHESIS: Anchoring a third attention-projection input column with equivalent full eight-coordinate initialization and implicit AdamW updates will reduce the model from 1,599 to 1,598 parameters while retaining at least 99% accuracy after 18,000 steps.

INTENDED_EDIT: Split a third projection column into seven learned relative coordinates plus a zero anchor, initialize all three anchored columns from full i.i.d. columns, optimize each with GaugeAdamW, and add 1,000 low-rate refinement steps.

EVIDENCE: Successive initialization-preserving projection-column anchoring achieved 99.98% at 1,600 parameters and 99.96% at 1,599 parameters, directly validating this exact symmetry and optimizer treatment for two columns.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.proj.weight = nn.Parameter(torch.empty(d_model, d_model - 2))
        self.proj_col_2 = nn.Parameter(torch.empty(d_model - 1))
        self.proj_col = nn.Parameter(torch.empty(d_model - 1))
=======
        self.proj = nn.Linear(d_model, d_model)
        self.proj.weight = nn.Parameter(torch.empty(d_model, d_model - 3))
        self.proj_col_3 = nn.Parameter(torch.empty(d_model - 1))
        self.proj_col_2 = nn.Parameter(torch.empty(d_model - 1))
        self.proj_col = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_col_2 = torch.cat(
            (self.proj_col_2, self.proj_col_2.new_zeros(1))
        ).unsqueeze(1)
        proj_col = torch.cat(
            (self.proj_col, self.proj_col.new_zeros(1))
        ).unsqueeze(1)
        proj_weight = torch.cat(
            (self.proj.weight, proj_col_2, proj_col), dim=1
        )
=======
        proj_col_3 = torch.cat(
            (self.proj_col_3, self.proj_col_3.new_zeros(1))
        ).unsqueeze(1)
        proj_col_2 = torch.cat(
            (self.proj_col_2, self.proj_col_2.new_zeros(1))
        ).unsqueeze(1)
        proj_col = torch.cat(
            (self.proj_col, self.proj_col.new_zeros(1))
        ).unsqueeze(1)
        proj_weight = torch.cat(
            (self.proj.weight, proj_col_3, proj_col_2, proj_col), dim=1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                for proj_col in (block.attn.proj_col_2, block.attn.proj_col):
=======
                for proj_col in (
                    block.attn.proj_col_3,
                    block.attn.proj_col_2,
                    block.attn.proj_col,
                ):
>>>>>>> REPLACE

<<<<<<< SEARCH
    ] + [
        block.attn.proj_col_2 for block in model.blocks
    ] + [
        block.attn.proj_col for block in model.blocks
=======
    ] + [
        block.attn.proj_col_3 for block in model.blocks
    ] + [
        block.attn.proj_col_2 for block in model.blocks
    ] + [
        block.attn.proj_col for block in model.blocks
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=17000)
=======
    p.add_argument("--train-steps", type=int, default=18000)
>>>>>>> REPLACE