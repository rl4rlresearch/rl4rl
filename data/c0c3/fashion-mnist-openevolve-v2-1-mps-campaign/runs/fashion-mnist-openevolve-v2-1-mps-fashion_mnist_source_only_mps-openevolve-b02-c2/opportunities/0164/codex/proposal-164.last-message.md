MECHANISM: Constant label smoothing with calibrated flip-ensemble sharpening

HYPOTHESIS: Restoring constant 0.02 label smoothing and temperature-0.80 evaluation will recover 9,328 correct predictions and validation cross-entropy near 0.19147.

INTENDED_EDIT: Train paired flip views with 0.02 label smoothing and divide confidence-adaptive fused log-probabilities by 0.80 during evaluation.

EVIDENCE: Reference Design 2 achieved the best available score, preserving 9,328 correct predictions while lowering cross-entropy to 0.191472; the current unsmoothed design reached only 9,320 correct.

<<<<<<< SEARCH
        return torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
=======
        fused_log_probabilities = torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
        return fused_log_probabilities / 0.80
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(model(paired_images), paired_labels)
=======
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
>>>>>>> REPLACE