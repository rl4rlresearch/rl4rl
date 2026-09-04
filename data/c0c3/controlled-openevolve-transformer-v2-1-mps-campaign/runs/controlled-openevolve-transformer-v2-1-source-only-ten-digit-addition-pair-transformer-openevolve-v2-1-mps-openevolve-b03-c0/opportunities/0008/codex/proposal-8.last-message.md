MECHANISM: Shared normalization scale across all LayerNorm sites

HYPOTHESIS: Tying `ln_f.weight` to the shared block LayerNorm weight will reduce deduplicated parameters from 1,620 to 1,612 while retaining at least 99% accuracy.

INTENDED_EDIT: Share both the final LayerNorm scale and bias with `blocks[0].ln1`, preserving all attention and MLP capacity.

EVIDENCE: Sharing the block LayerNorm scale and bias achieved 99.59% accuracy, and extending bias sharing to the final LayerNorm still achieved 99.40%; this motivates testing the remaining isolated eight-parameter normalization sharing constraint.

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.ln_f.bias = self.blocks[0].ln1.bias
=======
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.ln_f.weight = self.blocks[0].ln1.weight
        self.ln_f.bias = self.blocks[0].ln1.bias
>>>>>>> REPLACE