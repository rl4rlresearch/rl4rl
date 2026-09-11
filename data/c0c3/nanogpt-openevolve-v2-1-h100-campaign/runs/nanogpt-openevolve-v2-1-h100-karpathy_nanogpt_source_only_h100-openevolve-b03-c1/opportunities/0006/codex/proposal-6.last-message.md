MECHANISM: Later intermediate global mixing at constant compute

HYPOTHESIS: Moving the intermediate full-context layer from index 3 to index 4 while retaining six 512-token local layers and the final global layer will reduce val_bpb below 0.993365 without changing throughput materially.

INTENDED_EDIT: Use an explicit eight-layer SSSSLSSS pattern, placing full-context attention at layers 4 and 7.

EVIDENCE: Two global layers achieved the best result (0.993365); one global layer regressed to 0.994122 and three regressed to 0.994554, motivating a controlled placement ablation with the winning attention counts and window sizes unchanged.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSSLSSS" # full context at layers 4 and 7; S uses quarter context
>>>>>>> REPLACE