MECHANISM: Peak-augmented 56-unit full-resolution GRU

HYPOTHESIS: A 56-unit GRU retaining mean, maximum, and final recurrent features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 376,725,600 to approximately 334,084,800.

INTENDED_EDIT: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU hidden width from 68 to 56 units.

EVIDENCE: The qualified 60-unit design achieved 87.12% accuracy at 376,725,600 MACs, leaving a 2.12-point margin; every tested four-unit reduction from 92 through 60 units retained at least 85% accuracy.

<<<<<<< SEARCH
        self.hidden_size = 68
=======
        self.hidden_size = 56
>>>>>>> REPLACE