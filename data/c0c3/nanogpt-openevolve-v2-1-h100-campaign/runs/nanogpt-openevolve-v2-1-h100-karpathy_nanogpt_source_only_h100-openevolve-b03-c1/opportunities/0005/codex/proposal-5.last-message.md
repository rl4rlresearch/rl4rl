MECHANISM: Denser periodic global attention

HYPOTHESIS: Adding a third full-context layer at index 5 will lower val_bpb below 0.993365; the added global mixing should outweigh its small throughput cost.

INTENDED_EDIT: Change the repeating attention pattern from SSSL to SSL, producing full-context layers at indices 2, 5, and 7 while retaining 512-token local windows elsewhere.

EVIDENCE: Removing one global layer increased tokens only from 512.2M to 516.9M but worsened val_bpb from 0.993365 to 0.994122, indicating global mixing contributes more than its modest compute cost.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSL"  # sliding window pattern: L=full, S=quarter context
>>>>>>> REPLACE