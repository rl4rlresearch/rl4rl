MECHANISM: Higher-amplitude cosine learning-rate schedule

HYPOTHESIS: Raising the peak learning rate from 0.0025 to 0.003 will exceed 9,254 correct predictions by increasing finite-budget optimization progress while retaining decay to zero for stable convergence.

INTENDED_EDIT: Increase AdamW’s initial learning rate and the matching cosine schedule amplitude by 20%, leaving the proven architecture, loss, batch size, and strided EMA unchanged.

EVIDENCE: Successive batch-size reductions from 128 to 64 to 32 improved correct predictions from 9,210 to 9,235 to 9,250 at the same learning rate, indicating that greater optimization progress within the fixed exposure budget helps; a modest learning-rate increase tests this without adding steps or runtime-heavy operations.

<<<<<<< SEARCH
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
    return torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1e-4)
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 2.5e-3 * multiplier
=======
        group["lr"] = 3.0e-3 * multiplier
>>>>>>> REPLACE