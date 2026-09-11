MECHANISM: Incrementally stronger logit regularization

HYPOTHESIS: Lowering the logit softcap from 10 to 7.5 will improve `val_bpb` below `0.986439` without materially changing throughput or memory use.

INTENDED_EDIT: Apply a 7.5 tanh softcap while preserving the current architecture, optimizer, batching, schedule, and max-autotuned compilation.

EVIDENCE: Removing softcapping regressed `val_bpb` to `0.993511`, while strengthening the cap from 15 to 10 improved it from `0.986491` to `0.986439`; a moderate further reduction tests whether that directional benefit continues.

<<<<<<< SEARCH
        softcap = 10
=======
        softcap = 7.5
>>>>>>> REPLACE