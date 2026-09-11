MECHANISM: Faster second-moment adaptation

HYPOTHESIS: Reducing AdamW’s second-moment decay from 0.999 to 0.99 will exceed 9,254 correct predictions by adapting gradient scaling more quickly within the fixed 3,126-step training budget.

INTENDED_EDIT: Set AdamW betas explicitly to `(0.9, 0.99)` while preserving the architecture, learning-rate schedule, loss, batch size, and strided EMA.

EVIDENCE: Reducing batch size from 128 to 64 to 32 monotonically improved correct predictions from 9,210 to 9,235 to 9,250, indicating that finite-budget optimization benefits from more responsive updates; a shorter second-moment horizon tests that mechanism without adding computation.

<<<<<<< SEARCH
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
    return torch.optim.AdamW(
        model.parameters(), lr=2.5e-3, betas=(0.9, 0.99), weight_decay=1e-4
    )
>>>>>>> REPLACE