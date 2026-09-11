MECHANISM: Validated inference-aligned augmentation curriculum

HYPOTHESIS: Moving the broad-to-cardinal transition from 5/8 to 39/64 will improve validation_correct beyond 9,163 and reproduce the verified 9,167-result region.

INTENDED_EDIT: Use broad ±2 translations for the first 39/64 of training, then center/cardinal one-pixel translations, preserving all other settings.

EVIDENCE: Reference Design 1 achieved 9,167 correct and 0.23885 cross-entropy with this exact boundary, outperforming the otherwise identical current 5/8 design’s 9,163 correct and 0.24108 cross-entropy; later boundaries degraded monotonically.

<<<<<<< SEARCH
    if step * 8 < total_steps * 5:
=======
    if step * 64 < total_steps * 39:
>>>>>>> REPLACE