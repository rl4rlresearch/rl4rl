MECHANISM: Bias-free classification head

HYPOTHESIS: Removing the eight classifier bias parameters will retain at least 85% validation accuracy while preserving exact MACs and recurrent steps, improving the parameter tiebreaker.

INTENDED_EDIT: Disable the bias in the 98-to-8 classifier while leaving the verified recurrent architecture and training procedure unchanged.

EVIDENCE: The 98-unit model achieved 85.03% accuracy, whereas reducing recurrent width to 97 failed; a smaller head-only structural reduction is the lowest-risk next optimization.

<<<<<<< SEARCH
        self.classifier = nn.Linear(98, 8)
=======
        self.classifier = nn.Linear(98, 8, bias=False)
>>>>>>> REPLACE