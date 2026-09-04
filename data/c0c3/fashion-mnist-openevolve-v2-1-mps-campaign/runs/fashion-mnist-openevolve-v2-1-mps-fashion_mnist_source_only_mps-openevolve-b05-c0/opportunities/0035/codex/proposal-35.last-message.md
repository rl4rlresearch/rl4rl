MECHANISM: Increased per-update optimization distance

HYPOTHESIS: Raising the cosine schedule’s peak learning rate by 10% will exceed 9,254 correct predictions by capturing part of the optimization benefit previously observed from increasing optimizer-step count, without increasing runtime.

INTENDED_EDIT: Increase AdamW’s initial and scheduled base learning rate from 2.5e-3 to 2.75e-3 while preserving the proven architecture, loss, batch size, cosine decay, and strided EMA.

EVIDENCE: Reducing batch size from 128 to 64 to 32 monotonically improved correct predictions from 9,210 to 9,235 to 9,250, indicating that greater finite-budget optimization progress is beneficial; a modest learning-rate increase tests that mechanism without the timeout risk of another batch-size reduction.

<<<<<<< SEARCH
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
    return torch.optim.AdamW(model.parameters(), lr=2.75e-3, weight_decay=1e-4)
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 2.5e-3 * multiplier
=======
        group["lr"] = 2.75e-3 * multiplier
>>>>>>> REPLACE