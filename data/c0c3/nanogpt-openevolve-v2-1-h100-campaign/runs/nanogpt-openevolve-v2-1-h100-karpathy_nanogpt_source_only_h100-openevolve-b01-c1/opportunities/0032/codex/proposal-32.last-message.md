MECHANISM: Earlier two-stage full-context consolidation

HYPOTHESIS: Moving the intermediate full-context layer from layer 4 to layer 3 will lower val_bpb below 0.984467 by giving globally mixed features one additional local-refinement layer before the final full-context layer.

INTENDED_EDIT: Use full-context attention at layers 3 and 8 instead of layers 4 and 8, preserving parameter count, attention compute, batching, and optimization.

EVIDENCE: Layers 4 and 8 achieved 0.984467, while shifting the intermediate layer later to layer 5 regressed to 0.985851; testing the adjacent earlier placement probes the observed direction while holding the successful two-global-layer budget fixed.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # periodic full-context consolidation at layers 4 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # full-context consolidation at layers 3 and 8
>>>>>>> REPLACE