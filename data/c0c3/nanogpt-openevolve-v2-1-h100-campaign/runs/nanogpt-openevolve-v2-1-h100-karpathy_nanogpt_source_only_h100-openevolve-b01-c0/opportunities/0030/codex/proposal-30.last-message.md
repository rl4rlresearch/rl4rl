MECHANISM: Incremental global-attention density

HYPOTHESIS: Adding one mid-to-late full-attention layer will beat 0.984313 val_bpb because removing a full-attention layer degraded validation despite essentially unchanged token throughput.

INTENDED_EDIT: Retain the existing full-attention layers at depths 4 and 8 while converting depth 6 from short-window to full attention.

EVIDENCE: Changing `SSSL` to `SSSS` reduced full-attention layers from two to one and regressed val_bpb from 0.984313 to 0.985611 while processing nearly identical tokens (506.2M versus 505.7M), indicating that global-attention capacity—not throughput—was limiting that variant.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSLSLSL" # full attention at layers 4, 6, and 8
>>>>>>> REPLACE