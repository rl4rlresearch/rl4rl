MECHANISM: Complete LayerNorm affine sharing

HYPOTHESIS: Sharing the final LayerNorm gain with the already-shared block LayerNorm gain will reduce the model from 1,573 to 1,565 parameters while retaining at least 99% accuracy after 21,000 steps.

INTENDED_EDIT: Tie `ln_f.weight` to `blocks[0].ln1.weight`, completing the existing LayerNorm affine sharing without adding forward-pass or optimizer overhead.

EVIDENCE: The verified 1,573-parameter model reached 99.98% accuracy while already sharing the first and second block LayerNorm parameters and tying the final LayerNorm bias to them; recent one-parameter reconstructions repeatedly timed out, motivating a zero-overhead extension of the successful sharing pattern.

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.ln_f.bias = self.blocks[0].ln1.bias
=======
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.ln_f.weight = self.blocks[0].ln1.weight
        self.ln_f.bias = self.blocks[0].ln1.bias
>>>>>>> REPLACE