MECHANISM: Peak-augmented 72-unit full-resolution GRU

HYPOTHESIS: A 72-unit GRU retaining mean, maximum, and final recurrent features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 572,325,600 to approximately 519,670,080.

INTENDED_EDIT: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU from 76 to 72 hidden units.

EVIDENCE: The qualified 76-unit design achieved 86.13% accuracy at 572,325,600 MACs, leaving a 1.13-point margin and motivating the next controlled four-unit structural reduction.

<<<<<<< SEARCH
        self.hidden_size = 76
=======
        self.hidden_size = 72
>>>>>>> REPLACE