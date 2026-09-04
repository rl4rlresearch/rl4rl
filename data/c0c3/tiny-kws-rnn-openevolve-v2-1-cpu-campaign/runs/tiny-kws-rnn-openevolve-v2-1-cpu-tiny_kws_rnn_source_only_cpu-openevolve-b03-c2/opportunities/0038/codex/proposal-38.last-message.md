MECHANISM: Confidence-boundary bisection

HYPOTHESIS: A 0.7475 penultimate-step exit threshold will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 746,601,120 by admitting safe examples between the qualified 0.75 and failed 0.745 boundaries.

INTENDED_EDIT: Lower the qualified early-exit threshold from 0.80 to the midpoint 0.7475.

EVIDENCE: Threshold 0.75 qualified at 85.03% with 746,601,120 MACs, whereas 0.745 saved four additional recurrent steps but missed accuracy by one validation example; bisecting this narrow interval is the most informative next boundary test.

<<<<<<< SEARCH
        return penultimate & (confidence >= 0.80)
=======
        return penultimate & (confidence >= 0.7475)
>>>>>>> REPLACE