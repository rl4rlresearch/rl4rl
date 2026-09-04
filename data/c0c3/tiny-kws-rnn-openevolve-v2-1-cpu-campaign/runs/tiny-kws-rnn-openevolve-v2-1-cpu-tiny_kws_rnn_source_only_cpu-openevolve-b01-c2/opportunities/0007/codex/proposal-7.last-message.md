MECHANISM: Peak-augmented 80-unit full-resolution GRU

HYPOTHESIS: An 80-unit GRU retaining mean, maximum, and final-state features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 685,147,680 to approximately 627,484,800.

INTENDED_EDIT: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU from 92 to 80 hidden units.

EVIDENCE: The qualified 84-unit design achieved 87.73% accuracy at 685,147,680 MACs, leaving a 2.73-point margin and motivating another controlled four-unit structural reduction.

<<<<<<< SEARCH
        self.hidden_size = 92
=======
        self.hidden_size = 80
>>>>>>> REPLACE