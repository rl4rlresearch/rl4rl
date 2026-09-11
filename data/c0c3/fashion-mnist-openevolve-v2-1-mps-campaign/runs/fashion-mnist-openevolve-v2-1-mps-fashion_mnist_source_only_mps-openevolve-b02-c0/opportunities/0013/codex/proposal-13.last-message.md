MECHANISM: One-pixel-aligned translation augmentation

HYPOTHESIS: Restricting training translations from two pixels to one will exceed 9,157 correct predictions by concentrating the fixed exposure budget on the translation range that improved validation inference.

INTENDED_EDIT: Change random-crop augmentation from a five-by-five offset grid to a three-by-three offset grid while preserving the model, optimizer, batch size, and best inference ensemble.

EVIDENCE: One-pixel translation averaging improved validation_correct from 9,146 to 9,157, whereas adding two-pixel cardinal views reduced it to 9,149; this indicates that two-pixel shifts are less useful for the fixed validation distribution.

<<<<<<< SEARCH
    padding = 2
=======
    padding = 1
>>>>>>> REPLACE