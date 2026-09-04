MECHANISM: Reduced smoothing for information-constrained recurrence

HYPOTHESIS: The closest 20-step schedule, which reached 84.91% accuracy, will recover the single additional correct validation prediction needed to exceed 85% when label smoothing is reduced from 0.03 to 0.01, while retaining approximately 264,555,520 inference MACs.

INTENDED_EDIT: Use the previously tested 20-step schedule that additionally omits frames 1, 6, and 26 for 32-frame inputs, and reduce label smoothing to strengthen supervision without changing inference cost.

EVIDENCE: Omitting frame 26 produced the strongest failed 20-step result at 84.91%, substantially outperforming the frame-3 and frame-10 variants and missing qualification by only about 0.09 percentage points.

<<<<<<< SEARCH
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(15 * available_frames // 16)
            schedule = [index for index in schedule if index not in skipped]
=======
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(3 * available_frames // 16)
                skipped.add(13 * available_frames // 16)
                skipped.add(15 * available_frames // 16)
            if available_frames >= 32:
                skipped.add(available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(logits, labels, label_smoothing=0.03)
=======
    return F.cross_entropy(logits, labels, label_smoothing=0.01)
>>>>>>> REPLACE