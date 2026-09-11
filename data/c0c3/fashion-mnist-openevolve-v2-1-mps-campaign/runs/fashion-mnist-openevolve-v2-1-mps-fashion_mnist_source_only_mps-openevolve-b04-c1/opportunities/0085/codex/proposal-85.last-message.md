MECHANISM: Remove deterministic parameter shrinkage

HYPOTHESIS: Eliminating AdamW weight decay will exceed 9,328 correct predictions by allowing better late hard-label fitting after dropout and label smoothing anneal away.

INTENDED_EDIT: Set AdamW weight decay from 2e-4 to zero while preserving the architecture, learning-rate schedule, EMA, and stochastic regularization.

EVIDENCE: Increasing weight decay to 1e-2 reduced validation correct from 9,328 to 9,290, providing directional evidence that persistent shrinkage conflicts with this short fixed-exposure training regime.

<<<<<<< SEARCH
        weight_decay=2e-4,
=======
        weight_decay=0.0,
>>>>>>> REPLACE