MECHANISM: LayerNorm-null attention projection bias gauge

HYPOTHESIS: Centering the attention output-projection bias will reduce the model from 1621 to 1620 parameters while retaining at least 99% accuracy, because its removed featurewise-common component passes unchanged through the residual stream, is invisible to `ln2`, and is eliminated by the final LayerNorm.

INTENDED_EDIT: Replace the eight-parameter attention projection bias with seven learned centered contrasts while preserving its zero initialization and the original initialization RNG sequence.

EVIDENCE: Balanced common-bias gauges in both block LayerNorms and the terminal `fc2` path previously passed, whereas the failed 1620 embedding gauge substantially altered embedding optimization; this applies the proven local bias-centering mechanism to a distinct exact LayerNorm-null direction.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
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
        proj_bias = proj_bias - proj_bias.mean()
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE