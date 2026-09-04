MECHANISM: Expanded penultimate-frame confidence exit

HYPOTHESIS: Lowering the penultimate-step exit threshold from 0.90 to 0.85 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 753,120,312.

INTENDED_EDIT: Broaden the qualified penultimate-step early exit to examples with maximum softmax confidence of at least 0.85.

EVIDENCE: Lowering the threshold from 0.95 to 0.90 increased validation accuracy from 85.276% to 85.399% while reducing mean recurrent steps from 21.609 to 21.480 and total inference MACs by 4,688,460, supporting another incremental reduction.

<<<<<<< SEARCH
        return penultimate & (confidence >= 0.98)
=======
        return penultimate & (confidence >= 0.85)
>>>>>>> REPLACE