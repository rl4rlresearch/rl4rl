MECHANISM: 128-token local backbone with full-context consolidation

HYPOTHESIS: Halving the seven local-attention windows from 256 to 128 tokens will increase training throughput while the essential final full-context layer preserves global integration, lowering val_bpb below 0.992286.

INTENDED_EDIT: Use 128-token attention windows in every local layer while leaving the final layer’s forced 2048-token window unchanged.

EVIDENCE: Reducing local windows from 512 to 256 increased token throughput without degrading val_bpb, while making the final layer local regressed sharply to 1.015479; this motivates further shrinking only the local windows while retaining global consolidation.

<<<<<<< SEARCH
        short_window = long_window // 8
=======
        short_window = long_window // 16
>>>>>>> REPLACE