MECHANISM: Residual-stream output-bias gauge fixing

HYPOTHESIS: Fixing one attention output-bias coordinate to zero will reduce the model from 1609 to 1608 parameters while retaining at least 99% accuracy, because any all-ones component of this residual-branch bias is erased by subsequent LayerNorms.

INTENDED_EDIT: Replace the eight-coordinate attention projection bias with seven learned coordinates padded by one fixed zero, while leaving the verified two-coordinate query bias and three-coordinate value bias intact.

EVIDENCE: Reducing the query bias from two coordinates to one collapsed accuracy to 84.17%, so the next reduction should preserve that boundary; unlike the failed second-head key–value sharing, this edit removes only a residual-stream gauge degree of freedom without tying projection maps.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = self.proj(y) + F.pad(self.proj_bias, (0, 1))
        y = self.resid_drop(y)
>>>>>>> REPLACE