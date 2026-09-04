MECHANISM: Front-loaded context widening

HYPOTHESIS: Reallocating the redundant late 512-token window to layer 2 will lower val_bpb below 0.983718 by giving the first global-consolidation stage two consecutive wider-context layers whose features propagate through the remaining network.

INTENDED_EDIT: Change the window schedule from SSMLSSSL to SMMLSSSL, widening layers 2 and 3 to 512 tokens while retaining full-context attention at layers 4 and 8.

EVIDENCE: SSMLSSML achieved 0.983720, while removing its late medium window produced a marginally better 0.983718 with greater throughput, indicating that late widening is redundant and motivating reuse of that compute in the beneficial first stage.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSMLSSSL" # widen context only before the first global consolidation
=======
WINDOW_PATTERN = "SMMLSSSL" # concentrate both medium windows in the first consolidation stage
>>>>>>> REPLACE