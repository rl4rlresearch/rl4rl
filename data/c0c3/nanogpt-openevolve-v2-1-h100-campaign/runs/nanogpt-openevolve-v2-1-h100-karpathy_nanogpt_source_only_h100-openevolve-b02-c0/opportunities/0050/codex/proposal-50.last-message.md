MECHANISM: Shift intermediate global attention deeper

HYPOTHESIS: Moving the intermediate full-context layer from depth 4 to depth 5 will reduce val_bpb below 0.983635 by applying global mixing to more mature local features while retaining the empirically favored total of two global layers.

INTENDED_EDIT: Change the eight-layer attention pattern from SSS-L-SSS-L to SSSS-L-SS-L, preserving the 424-token local window and all other settings.

EVIDENCE: Removing the intermediate global layer regressed val_bpb to 0.984677, while adding a third global layer regressed it to 0.984553; this indicates that two global layers are preferable and motivates testing their placement instead of their count.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSSLSSL" # six 424-token local layers; full context at depths 5 and 8
>>>>>>> REPLACE