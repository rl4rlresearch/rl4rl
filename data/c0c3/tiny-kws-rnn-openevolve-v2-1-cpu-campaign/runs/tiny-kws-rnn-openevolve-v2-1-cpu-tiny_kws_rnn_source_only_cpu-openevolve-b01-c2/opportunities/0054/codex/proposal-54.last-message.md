MECHANISM: Incrementally relaxed confidence-gated penultimate exit

HYPOTHESIS: Lowering the qualified penultimate exit threshold from 0.75 to 0.73 will retain at least 85% validation accuracy while reducing total inference MACs below 251,773,740.

INTENDED_EDIT: Change only the posterior-confidence threshold to 0.73 so additional examples skip the final recurrent step.

EVIDENCE: Every tested threshold from 0.95 through 0.75 qualified and progressively reduced MACs; the 0.75 design achieved 85.15% accuracy at 251,773,740 MACs, motivating one further cautious 0.02 relaxation.

<<<<<<< SEARCH
        return (count == float(total_steps - 1)) & (confidence >= 0.79)
=======
        return (count == float(total_steps - 1)) & (confidence >= 0.73)
>>>>>>> REPLACE