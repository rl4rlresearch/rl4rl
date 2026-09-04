MECHANISM: Residual-stream bias gauge elimination

HYPOTHESIS: Fixing one attention output-projection bias coordinate at zero will reduce the model from 1,618 to 1,617 parameters while retaining at least 99% accuracy, because a uniform shift of that bias can be absorbed by the MLP output bias without changing the block’s function.

INTENDED_EDIT: Preserve the original projection-layer construction, store seven learned output biases, and reconstruct a fixed-zero first coordinate during attention.

EVIDENCE: The current one-key/one-value-bias design reached 99.96% at 1,618 parameters. Since additional QKV, positional-embedding, and MLP-input reductions failed, this tests a distinct exact one-parameter redundancy: LayerNorm makes uniform residual shifts invisible between the attention and MLP sublayers.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model)
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        proj_bias = torch.cat((self.proj.bias.new_zeros(1), self.proj.bias))
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE