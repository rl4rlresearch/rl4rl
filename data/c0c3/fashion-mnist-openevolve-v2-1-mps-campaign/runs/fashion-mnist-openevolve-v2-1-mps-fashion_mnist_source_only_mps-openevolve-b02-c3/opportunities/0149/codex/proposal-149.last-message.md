MECHANISM: Quadratically guided higher-order power-mean TTA

HYPOTHESIS: Raising the fusion order from 1.125 to 1.25 will preserve all 9,348 correct predictions while reducing validation cross-entropy below 0.18768770637512208.

INTENDED_EDIT: Increase only the axial-view probability power-mean order, retaining the proven crops, weights, training procedure, and temperature.

EVIDENCE: Moving below arithmetic fusion worsened cross-entropy to 0.18773939895629882, while moving equally far above it improved cross-entropy to 0.18768770637512208; those opposing probes imply the local optimum remains above 1.125 and approximately near 1.23.

<<<<<<< SEARCH
        fusion_power = 1.125
=======
        fusion_power = 1.25
>>>>>>> REPLACE