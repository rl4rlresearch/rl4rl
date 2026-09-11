MECHANISM: Peak-augmented 60-unit full-resolution GRU

HYPOTHESIS: A 60-unit GRU retaining mean, maximum, and final recurrent features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 421,870,080 to approximately 376,725,600.

INTENDED_EDIT: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU from 64 to 60 hidden units.

EVIDENCE: The qualified 64-unit design achieved 86.63% accuracy at 421,870,080 MACs, leaving a 1.63-point margin; every tested four-unit reduction from 92 through 64 units retained at least 85% accuracy.

<<<<<<< SEARCH
        self.hidden_size = 64
=======
        self.hidden_size = 60
>>>>>>> REPLACE