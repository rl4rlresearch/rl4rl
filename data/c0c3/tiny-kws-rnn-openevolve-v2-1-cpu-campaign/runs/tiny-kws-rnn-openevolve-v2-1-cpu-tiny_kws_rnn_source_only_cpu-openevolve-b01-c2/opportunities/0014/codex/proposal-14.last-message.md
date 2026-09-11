MECHANISM: Boundary-seeking peak-augmented 58-unit full-resolution GRU

HYPOTHESIS: A 58-unit GRU retaining all 32 frames plus mean, maximum, and final-state features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 376,725,600 to approximately 355,092,240.

INTENDED_EDIT: Narrow the recurrent hidden width from 72 to 58 units while preserving the proven full-resolution schedule, temporal summaries, and training procedure.

EVIDENCE: The 60-unit design achieved 87.12% accuracy, while 56 units fell to 83.93%; testing the midpoint is the most informative probe of the observed capacity boundary and offers lower MACs if it qualifies.

<<<<<<< SEARCH
        self.hidden_size = 72
=======
        self.hidden_size = 58
>>>>>>> REPLACE