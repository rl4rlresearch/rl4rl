MECHANISM: Expanded penultimate-frame confidence exit

HYPOTHESIS: Lowering the early-exit threshold from 0.95 to 0.90 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 757,808,772 by skipping the final recurrent step for more examples.

INTENDED_EDIT: Broaden the qualified penultimate-step early exit to examples with maximum softmax confidence of at least 0.90.

EVIDENCE: Lowering the threshold from 0.98 to 0.95 preserved validation accuracy exactly at 85.276% while reducing mean recurrent steps from 21.779 to 21.609 and total inference MACs by 6,206,628, supporting another incremental threshold reduction.

<<<<<<< SEARCH
        return penultimate & (confidence >= 0.95)
=======
        return penultimate & (confidence >= 0.90)
>>>>>>> REPLACE