MECHANISM: Delayed linear warmdown with full Muon refinement

HYPOTHESIS: Restoring five-step Muon orthogonalization and delaying linear warmdown until the final 40% will retain roughly 497M-token throughput while providing more full-rate optimization, reducing val_bpb below 0.995558.

INTENDED_EDIT: Restore five Muon polar iterations and shorten the existing linear warmdown from 50% to 40% of training time.

EVIDENCE: Five Muon iterations achieved 0.995558 versus 0.998073 for three iterations at identical 497.0M-token throughput; the cosine schedule fell to 413.7M tokens and 1.010979, motivating a duration-only adjustment to the proven linear schedule.

<<<<<<< SEARCH
                momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=weight_decay,
=======
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.4    # fraction of time budget for LR warmdown
>>>>>>> REPLACE