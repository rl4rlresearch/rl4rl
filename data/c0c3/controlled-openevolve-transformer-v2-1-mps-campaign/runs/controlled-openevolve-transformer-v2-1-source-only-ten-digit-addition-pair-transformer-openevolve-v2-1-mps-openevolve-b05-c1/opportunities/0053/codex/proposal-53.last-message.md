MECHANISM: Attention output-weight common-mode quotient

HYPOTHESIS: Fixing the eighth attention output-weight row to zero will reduce the model from 1,315 to 1,307 parameters while retaining at least 99% accuracy within 50,000 steps, because its input-dependent common-mode contribution is erased by downstream LayerNorms.

INTENDED_EDIT: Replace the 8×8 attention output projection with seven learned output rows, append a fixed-zero eighth coordinate, and retain the full eight-coordinate learned bias used by the query offset and MLP sharing.

EVIDENCE: The analogous MLP output-weight quotient reduced the verified model from 1,375 to 1,374 parameters and achieved 100% accuracy; unlike the failed positional quotient, this removes only an output common mode from a residual branch whose result is normalized downstream.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model - 1, bias=False)
        self.proj_bias = nn.Parameter(torch.zeros(d_model))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.proj.bias.mean()
=======
        q = q + self.proj_bias.mean()
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = F.pad(self.proj(y), (0, 1))
        y = y + self.proj_bias
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_biases = torch.stack(
            (self.attn.proj.bias.mean(), self.attn.proj.bias[0])
        )
=======
        shared_biases = torch.stack(
            (self.attn.proj_bias.mean(), self.attn.proj_bias[0])
        )
>>>>>>> REPLACE