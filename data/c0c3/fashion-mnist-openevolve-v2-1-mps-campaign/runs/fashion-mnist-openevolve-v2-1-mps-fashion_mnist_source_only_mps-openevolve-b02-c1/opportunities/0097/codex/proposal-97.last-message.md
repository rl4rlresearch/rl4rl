MECHANISM: Edge-replicated translation padding

HYPOTHESIS: Replacing normalized-zero translation borders with edge-replicated borders will achieve at least 9,287 validation-correct predictions without the overhead of per-image background estimation.

INTENDED_EDIT: Use replication padding for both training translations and evaluation TTA while preserving the architecture, optimizer, view weights, and temperature.

EVIDENCE: The background-aware padding hypothesis remained untested because verification timed out; replication padding addresses the same artificial-border issue with a native, lower-overhead operation.

<<<<<<< SEARCH
        padded = F.pad(images, (1, 1, 1, 1))
=======
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
>>>>>>> REPLACE

<<<<<<< SEARCH
    windows = F.pad(images, (1, 1, 1, 1)).unfold(2, 28, 1).unfold(3, 28, 1)
=======
    windows = F.pad(
        images, (1, 1, 1, 1), mode="replicate"
    ).unfold(2, 28, 1).unfold(3, 28, 1)
>>>>>>> REPLACE