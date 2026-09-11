MECHANISM: Additional late-intermediate global-context refresh

HYPOTHESIS: Adding one full-attention layer between the proven global layers will retain at least 440M training tokens and reduce val_bpb below 0.982905.

INTENDED_EDIT: Keep the verified-best 5× MLP, optimizer, batch, and schedule unchanged while changing the eight-layer attention pattern from global layers 3 and 7 to global layers 3, 5, and 7.

EVIDENCE: Removing the intermediate full-attention layer regressed val_bpb from 0.982905 to 0.983844 without improving measured throughput, showing that intermediate global-context mixing is valuable and motivating a controlled addition of one later global refresh.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSLSLSL" # full attention at layers 3, 5, and 7
>>>>>>> REPLACE