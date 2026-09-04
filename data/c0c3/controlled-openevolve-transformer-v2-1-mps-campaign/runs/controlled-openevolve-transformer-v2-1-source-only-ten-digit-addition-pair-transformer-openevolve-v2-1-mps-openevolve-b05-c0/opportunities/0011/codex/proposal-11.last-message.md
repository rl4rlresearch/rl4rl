MECHANISM: LayerNorm-invariant attention output-bias gauge fixing

HYPOTHESIS: Fixing one attention output-projection bias coordinate at zero will reduce the model from 1635 to 1634 parameters while retaining at least 99% accuracy, because any full bias can be shifted to this form by subtracting its final coordinate from every residual feature, a shift canceled by the downstream LayerNorms.

INTENDED_EDIT: Replace the eight-element attention output-projection bias with seven learned coordinates and one fixed zero, preserving all optimization-sensitive query and value biases.

EVIDENCE: The seven-coordinate value bias reached 99.32% at 1635 parameters, whereas removing a second value-bias coordinate collapsed to 9.51%; this motivates preserving that sensitive path and removing an exact gauge degree of freedom from the output bias instead.

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