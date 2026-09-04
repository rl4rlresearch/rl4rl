MECHANISM: Earlier global feature mixing

HYPOTHESIS: Moving the intermediate full-context layer from depth 4 to depth 3 will reduce val_bpb below 0.983635 by enabling global information exchange earlier, since moving it later to depth 5 sharply regressed validation quality.

INTENDED_EDIT: Change the eight-layer attention pattern from SSS-L-SSS-L to SS-L-SSSS-L while preserving the 424-token local window and all other settings.

EVIDENCE: The best design uses two global layers at depths 4 and 8 with val_bpb 0.983635, whereas shifting the intermediate global layer to depth 5 worsened val_bpb to 0.985418; testing the opposite direction is the most informative remaining placement change.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSLSSSSL" # full context at depths 3 and 8
>>>>>>> REPLACE