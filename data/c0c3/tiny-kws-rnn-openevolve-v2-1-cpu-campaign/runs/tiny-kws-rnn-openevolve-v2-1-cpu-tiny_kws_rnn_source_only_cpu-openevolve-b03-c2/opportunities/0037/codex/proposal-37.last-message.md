MECHANISM: Fine-grained confidence-boundary expansion

HYPOTHESIS: Lowering the penultimate-step exit threshold from 0.750 to 0.745 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 746,601,120.

INTENDED_EDIT: Admit a small additional band of high-confidence examples to the qualified penultimate-step early exit.

EVIDENCE: The 0.75 threshold qualified at 85.03% and reduced recurrent execution by 27 steps versus 0.80; the narrow 0.005 change seeks further savings while respecting the now-small accuracy margin.

<<<<<<< SEARCH
        return penultimate & (confidence >= 0.75)
=======
        return penultimate & (confidence >= 0.745)
>>>>>>> REPLACE