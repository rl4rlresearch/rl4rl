MECHANISM: Later two-stage full-context consolidation

HYPOTHESIS: Moving the intermediate full-context layer from layer 4 to layer 5 will lower val_bpb below 0.984467 by allowing an additional local layer to construct features before global mixing while retaining two local refinement layers before the final global layer.

INTENDED_EDIT: Use full-context attention at layers 5 and 8 instead of layers 4 and 8, preserving parameter count, attention-compute budget, batching, and optimization.

EVIDENCE: Full-context layers 4 and 8 improved val_bpb from 0.985229 to 0.984467, but adding layer 6 regressed to 0.985972 and distributing global heads across every layer regressed to 0.985719; this motivates holding the successful two-layer global-attention budget fixed while testing its placement.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # periodic full-context consolidation at layers 4 and 8
=======
WINDOW_PATTERN = "SSSSLSSL" # full-context consolidation at layers 5 and 8
>>>>>>> REPLACE