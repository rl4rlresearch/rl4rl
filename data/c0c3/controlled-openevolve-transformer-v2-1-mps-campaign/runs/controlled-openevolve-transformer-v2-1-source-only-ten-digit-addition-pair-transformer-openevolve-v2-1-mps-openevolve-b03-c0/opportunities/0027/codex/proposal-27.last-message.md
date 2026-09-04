MECHANISM: Zero-mean attention output-bias gauge

HYPOTHESIS: Reparameterizing the attention output bias in the zero-sum subspace will reduce the model from 1,614 to 1,613 parameters while retaining at least 99% accuracy, because uniform output-bias shifts are removed by subsequent LayerNorms and symmetric centering avoids the failed coordinate-anchored constraint.

INTENDED_EDIT: Store seven attention projection-bias parameters and reconstruct an eight-dimensional zero-mean bias during the learned attention forward pass.

EVIDENCE: Fixing one projection-bias coordinate at zero failed at 48.41%, despite the exact residual-shift redundancy; the current 1,614-parameter design reaches 99.91%, motivating a symmetry-preserving parameterization of that distinct redundancy.

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
        proj_bias = torch.cat((self.proj.bias, self.proj.bias.new_zeros(1)))
        proj_bias = proj_bias - proj_bias.mean()
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE