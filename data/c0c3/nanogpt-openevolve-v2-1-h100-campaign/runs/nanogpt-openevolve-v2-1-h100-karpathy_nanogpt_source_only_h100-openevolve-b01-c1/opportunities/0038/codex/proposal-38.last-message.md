MECHANISM: Fully widened first consolidation stage

HYPOTHESIS: Widening layer 1 to 512 tokens will lower val_bpb below 0.983689 by letting all three layers before the first full-context layer build wider-context features.

INTENDED_EDIT: Change the attention schedule from SMMLSSSL to MMMLSSSL, retaining full-context layers 4 and 8 while widening the remaining first-stage local layer.

EVIDENCE: Reallocating the late 512-token window to layer 2 improved val_bpb from 0.983718 to 0.983689, while late widening was redundant; extending the same front-loaded trend to layer 1 is the most direct test of whether first-stage context remains limiting.

<<<<<<< SEARCH
WINDOW_PATTERN = "SMMLSSSL" # concentrate both medium windows in the first consolidation stage
=======
WINDOW_PATTERN = "MMMLSSSL" # widen every local layer before the first global consolidation
>>>>>>> REPLACE