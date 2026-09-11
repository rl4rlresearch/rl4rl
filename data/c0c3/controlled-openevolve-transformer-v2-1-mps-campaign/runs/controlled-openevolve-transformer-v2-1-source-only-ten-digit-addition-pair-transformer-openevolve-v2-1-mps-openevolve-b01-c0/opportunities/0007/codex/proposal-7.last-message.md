MECHANISM: Granular output-bias ablation with preserved value-bias pathway

HYPOTHESIS: Removing one attention output-projection bias scalar will reduce the model to 1635 parameters while retaining at least 99% accuracy, because the full learned value bias remains available to represent the missing output offset.

INTENDED_EDIT: Replace the eight-element projection bias with seven learned elements plus one fixed zero while preserving constructor RNG consumption.

EVIDENCE: The 1636-parameter model achieved 99.98%, whereas removing or tying a value-bias coordinate failed sharply; this motivates reducing the redundant projection-bias side while leaving every optimization-sensitive value-bias coordinate intact.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        # Construct the original projection first to preserve constructor RNG
        # consumption, then retain seven of its eight learned bias coordinates.
        self.proj = nn.Linear(d_model, d_model)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj.bias = None
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        proj_bias = torch.cat((self.proj_bias, self.proj_bias.new_zeros(1)))
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE