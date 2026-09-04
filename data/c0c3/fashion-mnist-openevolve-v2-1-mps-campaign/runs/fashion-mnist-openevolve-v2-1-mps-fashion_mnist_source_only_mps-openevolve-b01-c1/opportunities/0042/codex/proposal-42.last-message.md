MECHANISM: Cosine-decayed label smoothing

HYPOTHESIS: Decaying label smoothing from 0.04 to 0.00 while preserving its mean at 0.02 will exceed 9,239 correct predictions by emphasizing regularized feature learning early and sharper decision boundaries late.

INTENDED_EDIT: Apply the existing cosine training progress to label smoothing in all individual-view and ensemble cross-entropy terms, leaving architecture, compute, optimizer, and ensemble supervision unchanged.

EVIDENCE: The cosine ensemble-loss curriculum improved correctness from 9,237 to 9,239 while architecture and head changes regressed, motivating another compute-neutral temporal allocation of an established loss component.

<<<<<<< SEARCH
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    offsets = torch.randint(0, 5, (2,))
=======
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    label_smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
    offsets = torch.randint(0, 5, (2,))
>>>>>>> REPLACE

<<<<<<< SEARCH
        labels.repeat(4),
        label_smoothing=0.02,
=======
        labels.repeat(4),
        label_smoothing=label_smoothing,
>>>>>>> REPLACE

<<<<<<< SEARCH
        labels.repeat(2),
        label_smoothing=0.02,
=======
        labels.repeat(2),
        label_smoothing=label_smoothing,
>>>>>>> REPLACE

<<<<<<< SEARCH
        labels,
        label_smoothing=0.02,
=======
        labels,
        label_smoothing=label_smoothing,
>>>>>>> REPLACE