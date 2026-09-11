MECHANISM: Validated 39/64 augmentation curriculum

HYPOTHESIS: Starting center/cardinal augmentation after 39/64 of training will exceed the current 9,155 correct predictions and reproduce the best observed 9,167-result region.

INTENDED_EDIT: Replace the current two-thirds broad-translation phase with the verified 39/64 transition while preserving every other setting.

EVIDENCE: The otherwise identical 39/64 reference achieved 9,167 correct with 0.23885 cross-entropy, outperforming both the current two-thirds design’s 9,155 correct and the five-eighths design’s 9,163 correct.

<<<<<<< SEARCH
    if step * 3 < total_steps * 2:
=======
    if step * 64 < total_steps * 39:
>>>>>>> REPLACE