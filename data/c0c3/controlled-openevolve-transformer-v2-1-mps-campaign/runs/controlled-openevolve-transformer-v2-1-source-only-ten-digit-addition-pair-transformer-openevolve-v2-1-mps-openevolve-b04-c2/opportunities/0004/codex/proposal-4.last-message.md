MECHANISM: Absorbable pre-MLP normalization offset removal

HYPOTHESIS: Removing only the eight-parameter `ln2` bias will reduce the model from 1,644 to 1,636 parameters while retaining at least 99% accuracy, because its effect can be represented by the immediately following learned `fc1` bias.

INTENDED_EDIT: Disable the bias in the LayerNorm feeding the MLP while preserving all projection biases, widths, training settings, and decoding behavior.

EVIDENCE: Projection-bias removal and feed-forward width reduction failed at 72.22% and 74.39%; this instead preserves those parameters and removes an algebraically redundant normalization offset.

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE