MECHANISM: Three-stage full-context consolidation

HYPOTHESIS: Adding full-context attention at layer 6 while retaining it at layers 4 and 8 will reduce val_bpb below 0.984467, despite lower token throughput, by letting later layers reintegrate global information before final prediction.

INTENDED_EDIT: Change the attention schedule from full-context layers 4 and 8 to layers 4, 6, and 8; all other architecture and optimization settings remain unchanged.

EVIDENCE: Moving from one full-context layer to layers 4 and 8 improved val_bpb from 0.985229 to 0.984467 even as total tokens fell, while making attention entirely local regressed sharply to 1.015479; this motivates a controlled increase in global-attention depth.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # periodic full-context consolidation at layers 4 and 8
=======
WINDOW_PATTERN = "SSSLSLSS" # full-context consolidation at layers 4, 6, and 8
>>>>>>> REPLACE