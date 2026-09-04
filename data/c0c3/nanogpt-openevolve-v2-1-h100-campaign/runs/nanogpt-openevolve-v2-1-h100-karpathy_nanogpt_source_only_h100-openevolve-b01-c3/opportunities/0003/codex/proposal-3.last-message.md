MECHANISM: Sparse full-context attention allocation

HYPOTHESIS: Using half-context attention in the intermediate fourth layer while retaining full attention in the final layer will increase token throughput and reduce val_bpb below 0.995558.

INTENDED_EDIT: Change the repeating attention pattern from SSSL to SSSS; the existing forced-full final layer still guarantees one full-context layer.

EVIDENCE: The 262K-token batch processed only 393.2M tokens and worsened val_bpb to 0.998629 versus 497.0M tokens and 0.995558 for the current design, motivating a low-risk reduction in attention compute without increasing optimizer frequency.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSS" # half context throughout, with the final layer forced to full context
>>>>>>> REPLACE