MECHANISM: Targeted full-context attention densification

HYPOTHESIS: Adding one mid-late full-context layer while restoring the proven 524,288-token batch will retain near-497M-token throughput and reduce val_bpb below 0.995558.

INTENDED_EDIT: Restore the best-performing global batch and expand the attention pattern from two to three full-context layers, placing the added full layer at depth 5.

EVIDENCE: Removing the intermediate full-context layer changed throughput negligibly (497.0M to 496.5M tokens) but worsened val_bpb from 0.995558 to 0.997009, indicating that another strategically placed full-context layer may improve quality at low throughput cost.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSLSLSS" # full context at layers 3, 5, and the forced-full final layer
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
>>>>>>> REPLACE