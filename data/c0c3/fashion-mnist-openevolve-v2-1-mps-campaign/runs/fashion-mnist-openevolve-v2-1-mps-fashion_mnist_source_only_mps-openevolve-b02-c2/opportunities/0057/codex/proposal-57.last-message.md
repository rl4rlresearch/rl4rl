MECHANISM: Example-time-matched large-batch AdamW

HYPOTHESIS: Using 200-example batches with proportionally scaled learning rate and moment decay will exceed 9,322 correct predictions while finishing reliably by reducing training from 782 to 500 uniform optimizer steps.

INTENDED_EDIT: Preserve the best top-four attention model and paired-view loss, increase batch size to 200, and rescale AdamW’s learning rate and betas to approximately preserve optimizer dynamics per example.

EVIDENCE: The current top-four model achieved the best result at 9,322 correct but required 78.8 seconds, while numerous subsequent 782–800-step trials timed out; batch size 200 exactly divides 50,000 and removes 282 optimizer steps without discarding examples.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 200
>>>>>>> REPLACE

<<<<<<< SEARCH
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
=======
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.90625e-3,
        weight_decay=5e-4,
        betas=(0.85, 0.984),
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 2.5e-3 * multiplier
=======
        group["lr"] = 3.90625e-3 * multiplier
>>>>>>> REPLACE