MECHANISM: Peak-augmented 64-unit full-resolution GRU

HYPOTHESIS: A 64-unit GRU retaining mean, maximum, and final recurrent features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 469,518,240 to approximately 421,870,080.

INTENDED_EDIT: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU from 84 to 64 hidden units.

EVIDENCE: The qualified 68-unit design achieved 86.87% accuracy at 469,518,240 MACs, leaving a 1.87-point margin and motivating the next controlled four-unit structural reduction.

<<<<<<< SEARCH
        self.hidden_size = 84
=======
        self.hidden_size = 64
>>>>>>> REPLACE