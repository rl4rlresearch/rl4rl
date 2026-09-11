MECHANISM: Intermediate local-context expansion

HYPOTHESIS: Increasing short-attention windows from 512 to 768 tokens will beat 0.984244 val_bpb by adding mid-range context without the throughput and optimization cost of a fourth full-attention layer.

INTENDED_EDIT: Expand every short-attention layer to three-eighths of the 2048-token sequence while retaining full attention at layers 4, 6, and 8.

EVIDENCE: Three full-attention layers improved val_bpb to 0.984244 despite lower throughput, but four full-attention layers regressed to 0.985402; partially expanding the remaining local layers tests an intermediate context-compute point.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = 3 * long_window // 8
>>>>>>> REPLACE