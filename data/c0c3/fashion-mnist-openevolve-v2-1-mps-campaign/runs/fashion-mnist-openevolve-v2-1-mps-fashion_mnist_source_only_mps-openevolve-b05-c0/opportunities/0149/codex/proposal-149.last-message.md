MECHANISM: Cosine label-smoothing curriculum

HYPOTHESIS: Annealing label smoothing from 0.07 to 0.03 while retaining a 0.05 temporal average will exceed 9,260 correct predictions by combining stronger early regularization with stronger late class separation.

INTENDED_EDIT: Replace fixed 0.05 label smoothing with a compute-neutral cosine schedule from 0.07 at initialization to 0.03 at the final training step.

EVIDENCE: Fixed 0.03 and 0.07 smoothing reduced validation_correct to 9,242 and 9,244 respectively, while fixed 0.05 achieved 9,260; scheduling symmetrically around 0.05 preserves the verified average strength while testing whether regularization timing, rather than its overall amount, improves the short two-pass optimization.

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    progress = step / max(total_steps - 1, 1)
    label_smoothing = 0.05 + 0.02 * math.cos(math.pi * progress)
    return F.cross_entropy(
        model(images), labels, label_smoothing=label_smoothing
    )
>>>>>>> REPLACE