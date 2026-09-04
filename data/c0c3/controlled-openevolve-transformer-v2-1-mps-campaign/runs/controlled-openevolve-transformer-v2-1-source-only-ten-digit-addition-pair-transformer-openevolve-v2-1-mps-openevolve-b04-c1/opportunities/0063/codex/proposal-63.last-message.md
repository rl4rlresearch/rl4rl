MECHANISM: Absorbable MLP LayerNorm scale elimination

HYPOTHESIS: Removing the eight learned `ln2` scale parameters will reduce the model from 1445 to 1437 parameters while retaining at least 99% accuracy, because `fc1` can absorb any diagonal LayerNorm scaling and initialization remains functionally identical.

INTENDED_EDIT: Make the MLP-input LayerNorm non-affine, eliminating eight redundant trainable scales without changing the initial model function or random stream.

EVIDENCE: The 1445-parameter design achieved 99.91% accuracy after extensive symmetry quotienting; unlike earlier one-dimensional gauges, all eight `ln2` scales are algebraically redundant with the immediately following dense `fc1` weights.

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE