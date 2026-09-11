MECHANISM: Single global-attention consolidation layer

HYPOTHESIS: Using 384-token local attention in the first seven layers and retaining full context only in the final layer will reduce `val_bpb` below 0.983618 by increasing token exposure while preserving end-to-end access to the entire sequence.

INTENDED_EDIT: Restore the best verified 35% AdamW cooldown and 384-token local window, then remove the intermediate full-context layer by changing the window pattern to all-short; the existing final-layer override remains full-context.

EVIDENCE: The 384-token design achieved the best result, `val_bpb` 0.983618 at 510.9M tokens, outperforming both 512-token and 256-token windows. Because seven stacked 384-token causal layers have a receptive field exceeding 2048 tokens and the final layer remains globally attentive, the intermediate global layer may be redundant compute.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = 3 * long_window // 16
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSS" # 384-token local layers; final layer is forced full-context
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.40   # test a shorter AdamW cooldown to preserve late refinement updates
=======
WARMDOWN_RATIO = 0.35   # best verified AdamW cooldown
>>>>>>> REPLACE