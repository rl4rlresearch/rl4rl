MECHANISM: Fine-bracket full-state exponential weight consolidation

HYPOTHESIS: A 0.98125-decay full-state EMA will exceed the current validation_score by selecting an averaging horizon between the two 9,344-correct endpoints, potentially improving correctness or tie-breaking cross-entropy.

INTENDED_EDIT: Retain the proven architecture and training procedure while changing only the EMA decay from 0.98 to 0.98125.

EVIDENCE: Decays 0.98 and 0.9825 both achieved the best observed 9,344 correct, while neighboring 0.975 and 0.985 achieved 9,340 and 9,343; testing their midpoint is the most focused refinement of the observed optimum plateau.

<<<<<<< SEARCH
                decay = 0.98 ** elapsed
=======
                decay = 0.98125 ** elapsed
>>>>>>> REPLACE