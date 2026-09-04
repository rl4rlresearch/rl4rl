MECHANISM: Absorbable attention LayerNorm scale elimination

HYPOTHESIS: Making `ln1` non-affine will reduce the model from 1437 to 1430 learned parameters while retaining at least 99% accuracy, because the following `qkv` projection can absorb any diagonal LayerNorm scaling and initialization remains functionally identical.

INTENDED_EDIT: Remove the seven learned `ln1` scale coordinates by replacing `ScaleFixedLayerNorm` with non-affine LayerNorm.

EVIDENCE: The analogous removal of all eight `ln2` scales retained 99.93% accuracy at 1437 parameters; `ln1` has the same absorbable-scale relationship with its immediately following dense `qkv` projection.

<<<<<<< SEARCH
        self.ln1 = ScaleFixedLayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE