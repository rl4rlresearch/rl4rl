MECHANISM: Periodic full-context consolidation

HYPOTHESIS: Adding an intermediate full-context attention layer will lower val_bpb below 0.985229 by removing the single-final-layer global-information bottleneck, despite modestly reduced token throughput.

INTENDED_EDIT: Change the attention pattern from seven local layers plus one forced full-context layer to six local layers and full-context attention at layers 4 and 8.

EVIDENCE: Making the final layer local regressed sharply to 1.015479, showing that global integration is disproportionately valuable, while throughput-oriented 128-token windows and grouped-query attention also regressed; this motivates testing additional global capacity rather than pursuing throughput alone.

<<<<<<< SEARCH
WINDOW_PATTERN = "S"    # all local attention; final layer is forced to full context
=======
WINDOW_PATTERN = "SSSL" # periodic full-context consolidation at layers 4 and 8
>>>>>>> REPLACE