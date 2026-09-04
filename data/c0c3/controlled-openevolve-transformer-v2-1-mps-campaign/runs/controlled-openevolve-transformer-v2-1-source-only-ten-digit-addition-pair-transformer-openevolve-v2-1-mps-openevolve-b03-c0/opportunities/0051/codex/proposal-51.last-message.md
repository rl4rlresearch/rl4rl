MECHANISM: Initialization-preserving attention-output column gauge

HYPOTHESIS: Anchoring one uniform-output gauge coordinate in an attention-projection weight column, while preserving equivalent full initialization and implicit eight-coordinate AdamW updates, will reduce the model from 1,601 to 1,600 parameters and retain at least 99% accuracy after 16,000 steps.

INTENDED_EDIT: Split the final attention-projection input column into seven learned relative coordinates plus a zero anchor, initialize it from an eight-coordinate i.i.d. column, optimize it with GaugeAdamW, and add 1,000 low-rate refinement steps.

EVIDENCE: The earlier projection-column gauge reached only 40.05%, but the later initialization-preserving token-embedding gauge achieved 99.99% at 1,601 parameters; this tests whether transferring that validated full-initialization treatment rescues the exact uniform-output projection gauge.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model)
        self.proj.weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.proj_col = nn.Parameter(torch.empty(d_model - 1))
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_bias = torch.cat((self.proj.bias, self.proj.bias.new_zeros(1)))
        y = F.linear(y, self.proj.weight, proj_bias)
=======
        proj_col = torch.cat(
            (self.proj_col, self.proj_col.new_zeros(1))
        ).unsqueeze(1)
        proj_weight = torch.cat((self.proj.weight, proj_col), dim=1)
        proj_bias = torch.cat((self.proj.bias, self.proj.bias.new_zeros(1)))
        y = F.linear(y, proj_weight, proj_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
        with torch.no_grad():
            full_token_weight = self.token_emb.weight.new_empty(
=======
        self.apply(self._init_weights)
        with torch.no_grad():
            for block in self.blocks:
                full_proj_col = block.attn.proj_col.new_empty(cfg.d_model)
                nn.init.normal_(full_proj_col, mean=0.0, std=0.02)
                full_proj_col.sub_(full_proj_col[-1].clone())
                block.attn.proj_col.copy_(full_proj_col[:-1])

            full_token_weight = self.token_emb.weight.new_empty(
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [model.token_emb.weight] + [
        block.attn.proj.bias for block in model.blocks
    ]
=======
    gauge_parameters = [model.token_emb.weight] + [
        block.attn.proj.bias for block in model.blocks
    ] + [
        block.attn.proj_col for block in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=15000)
=======
    p.add_argument("--train-steps", type=int, default=16000)
>>>>>>> REPLACE