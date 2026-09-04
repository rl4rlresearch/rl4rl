MECHANISM: Peak-augmented 84-unit full-resolution GRU

HYPOTHESIS: An 84-unit GRU retaining mean, maximum, and final-state features will achieve at least 85% validation accuracy while reducing total inference MACs from 745,314,240 to approximately 685,147,680.

INTENDED_EDIT: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU from 88 to 84 hidden units.

EVIDENCE: The 88-unit design achieved 88.47% accuracy—3.47 points above the requirement—at 745,314,240 MACs, supporting another controlled four-unit structural reduction without discarding temporal observations.

<<<<<<< SEARCH
        self.hidden_size = 88
=======
        self.hidden_size = 84
>>>>>>> REPLACE