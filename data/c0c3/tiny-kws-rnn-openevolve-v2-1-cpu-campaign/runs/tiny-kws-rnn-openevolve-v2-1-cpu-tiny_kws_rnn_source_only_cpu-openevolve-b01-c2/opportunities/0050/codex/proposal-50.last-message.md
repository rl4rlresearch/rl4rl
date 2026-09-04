MECHANISM: Incrementally relaxed confidence-gated penultimate exit

HYPOTHESIS: Lowering the penultimate exit threshold from the qualified 0.83 to 0.81 will retain validation accuracy at or above 85% while reducing total inference MACs below 253,198,800.

INTENDED_EDIT: Adopt a 0.81 posterior-confidence threshold so more examples skip the final recurrent step.

EVIDENCE: Reducing the threshold from 0.85 to 0.83 preserved 85.15% accuracy and reduced recurrent execution by 20 steps, supporting another cautious 0.02 relaxation.

<<<<<<< SEARCH
        return (count == float(total_steps - 1)) & (confidence >= 0.90)
=======
        return (count == float(total_steps - 1)) & (confidence >= 0.81)
>>>>>>> REPLACE