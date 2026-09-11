MECHANISM: Single-coordinate first-pre-norm scale absorption

HYPOTHESIS: Fixing one `ln1` scale to one will reduce the qualified model from 1,539 to 1,538 parameters while retaining at least 99% accuracy, because the corresponding columns of the learned query, key, and value projections can independently absorb that scale.

INTENDED_EDIT: Reuse the qualified seven-scale bias-free LayerNorm for `ln1`, while preserving the current attention, MLP, gauges, optimizer, and decoding behavior.

EVIDENCE: Fixing one `ln2` scale achieved 99.93% accuracy at 1,539 parameters, while removing all eight scales failed; applying the same minimal one-coordinate constraint to the distinct `ln1` scale vector is the most controlled untested reduction.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln1 = OneFixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE