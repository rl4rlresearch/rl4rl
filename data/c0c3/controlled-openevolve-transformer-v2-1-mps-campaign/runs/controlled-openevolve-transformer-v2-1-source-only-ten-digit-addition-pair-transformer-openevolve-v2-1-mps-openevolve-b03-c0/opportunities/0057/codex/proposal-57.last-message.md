MECHANISM: Gauge-anchored MLP output bias

HYPOTHESIS: Anchoring the uniform-shift coordinate of the final MLP bias will reduce the model from 1,597 to 1,596 parameters while retaining at least 99% accuracy after 20,000 steps, because this shift is removed by the final LayerNorm.

INTENDED_EDIT: Store seven relative `fc2` bias coordinates, reconstruct an eighth zero anchor, train them with implicit eight-coordinate GaugeAdamW, and add 1,000 low-rate refinement steps.

EVIDENCE: Gauge-aware anchoring of the analogous attention-projection bias retained 99.89% accuracy, while the current 1,597-parameter design reached 99.97%; this tests the same validated uniform-output symmetry in the MLP instead of repeating the failed third positional-row anchor.

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        return self.drop(self.fc2(F.gelu(hidden)))
=======
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        fc2_bias = torch.cat((self.fc2.bias, self.fc2.bias.new_zeros(1)))
        output = F.linear(F.gelu(hidden), self.fc2.weight, fc2_bias)
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
    ] + [
        block.attn.proj.bias for block in model.blocks
    ] + [
        block.attn.proj_col_2 for block in model.blocks
=======
    ] + [
        block.attn.proj.bias for block in model.blocks
    ] + [
        block.mlp.fc2.bias for block in model.blocks
    ] + [
        block.attn.proj_col_2 for block in model.blocks
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=19000)
=======
    p.add_argument("--train-steps", type=int, default=20000)
>>>>>>> REPLACE