MECHANISM: Peak-augmented 68-unit full-resolution GRU

HYPOTHESIS: A 68-unit GRU retaining mean, maximum, and final recurrent features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 519,670,080 to approximately 469,518,240.

INTENDED_EDIT: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU from 80 to 68 hidden units.

EVIDENCE: The qualified 72-unit design achieved 86.63% accuracy at 519,670,080 MACs, leaving a 1.63-point margin and motivating the next controlled four-unit width reduction.

<<<<<<< SEARCH
        self.hidden_size = 80
=======
        self.hidden_size = 68
>>>>>>> REPLACE