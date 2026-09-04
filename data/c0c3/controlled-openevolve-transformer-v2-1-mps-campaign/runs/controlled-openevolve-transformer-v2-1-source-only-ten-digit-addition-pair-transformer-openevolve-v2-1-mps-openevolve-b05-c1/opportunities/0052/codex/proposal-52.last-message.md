MECHANISM: Third cross-branch bias-coordinate sharing

HYPOTHESIS: Reusing `attn.proj.bias[1]` as a third MLP output-bias coordinate will reduce the model from 1,315 to 1,314 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Reduce the dedicated MLP output bias from five coordinates to four and append three learned coordinates derived from the attention projection bias, preserving the seven-coordinate MLP output bias, zero initialization, and all training settings.

EVIDENCE: The current 1,315-parameter design achieved 100% accuracy after sharing both the attention projection-bias mean and `attn.proj.bias[0]` with the MLP; extending that successful mechanism by one adjacent bias coordinate is the smallest informative reduction.

<<<<<<< SEARCH
        self.output_bias = nn.Parameter(torch.zeros(d_model - 3))
=======
        self.output_bias = nn.Parameter(torch.zeros(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_biases = torch.stack(
            (self.attn.proj.bias.mean(), self.attn.proj.bias[0])
        )
=======
        shared_biases = torch.stack(
            (
                self.attn.proj.bias.mean(),
                self.attn.proj.bias[0],
                self.attn.proj.bias[1],
            )
        )
>>>>>>> REPLACE