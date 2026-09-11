MECHANISM: Earlier global feature mixing

HYPOTHESIS: Moving the intermediate full-context layer from depth 4 to depth 3 will reduce val_bpb below 0.983635 by enabling global information exchange earlier.

INTENDED_EDIT: Change the attention pattern from SSS-L-SSS-L to SS-L-SSSS-L while retaining 424-token local windows and all other settings.

EVIDENCE: Shifting the intermediate global layer later to depth 5 worsened val_bpb from 0.983635 to 0.985418; the earlier-placement attempt produced no verifiable implementation, so its quality remains untested.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSLSSSSL" # global attention at layers 3 and 8
>>>>>>> REPLACE