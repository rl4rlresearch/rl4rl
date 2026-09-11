MECHANISM: Cosine label-smoothing handoff

HYPOTHESIS: Cosine-annealing label smoothing during the second half will exceed 9,328 correct predictions by preserving useful smoothing early in the annealing phase while removing soft-target bias more quickly near the EMA-dominated endpoint.

INTENDED_EDIT: Keep dropout’s linear decay unchanged, but replace label smoothing’s linear decay with a cosine schedule from 0.02 to zero.

EVIDENCE: Ending smoothing earlier reduced validation correct to 9,316, while retaining more smoothing reached only 9,325; a cosine handoff targets the useful middle ground without adding runtime.

<<<<<<< SEARCH
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    return F.cross_entropy(
=======
    logits = model(images)
    smoothing_decay = 0.5 * (
        1.0 - math.cos(math.pi * dropout_decay)
    )
    label_smoothing = 0.02 * (1.0 - smoothing_decay)
    return F.cross_entropy(
>>>>>>> REPLACE