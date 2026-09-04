MECHANISM: Incrementally relaxed confidence-gated penultimate exit

HYPOTHESIS: Lowering the qualified penultimate exit threshold from 0.77 to 0.75 will retain validation accuracy at or above 85% while reducing total inference MACs below 252,165,240.

INTENDED_EDIT: Exit additional examples after recurrent step 21 by changing only the posterior-confidence threshold to 0.75.

EVIDENCE: Every tested threshold from 0.95 through 0.77 qualified and progressively reduced MACs; the 0.77 design achieved 85.15% accuracy at 252,165,240 MACs, motivating one more cautious 0.02 relaxation.

<<<<<<< SEARCH
        return (count == float(total_steps - 1)) & (confidence >= 0.77)
=======
        return (count == float(total_steps - 1)) & (confidence >= 0.75)
>>>>>>> REPLACE