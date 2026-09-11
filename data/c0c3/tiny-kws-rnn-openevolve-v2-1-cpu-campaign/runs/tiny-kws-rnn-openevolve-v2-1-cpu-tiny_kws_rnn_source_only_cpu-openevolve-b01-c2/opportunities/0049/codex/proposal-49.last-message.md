MECHANISM: Incrementally relaxed confidence-gated penultimate exit

HYPOTHESIS: Lowering the penultimate exit threshold from 0.85 to 0.83 will retain validation accuracy at or above 85% while reducing total inference MACs below 253,512,000.

INTENDED_EDIT: Exit additional high-confidence examples after step 21 by changing only the posterior-confidence threshold to 0.83.

EVIDENCE: Threshold reductions from 0.95 to 0.90 and then 0.85 consistently reduced MACs while remaining qualified; the 0.85 design achieved 85.15% accuracy and 253,512,000 MACs, supporting a smaller 0.02 relaxation given its narrow accuracy margin.

<<<<<<< SEARCH
        return (count == float(total_steps - 1)) & (confidence >= 0.85)
=======
        return (count == float(total_steps - 1)) & (confidence >= 0.83)
>>>>>>> REPLACE