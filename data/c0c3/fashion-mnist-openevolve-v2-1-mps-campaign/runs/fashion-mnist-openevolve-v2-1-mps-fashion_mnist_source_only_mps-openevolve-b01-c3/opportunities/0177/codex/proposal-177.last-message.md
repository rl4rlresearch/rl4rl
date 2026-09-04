MECHANISM: Late-ramped flip-offset orbit exposure

HYPOTHESIS: Ramping cross-offset pairs from 25% to 50% while preserving the validated 37.5% average exposure will exceed 9,325 correct predictions by emphasizing exact-reflection learning early and translation invariance after class features stabilize.

INTENDED_EDIT: Replace the fixed 37.5% cross-offset probability with a linear 25%-to-50% training schedule; preserve architecture, loss strength, compute, and inference pooling.

EVIDENCE: Raising fixed cross-offset exposure from 25% to 37.5% improved validation correctness from 9,318 to 9,325, establishing a beneficial dose response; this tests whether scheduling the same average dose improves when that invariance is learned.

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    batch, channels, height, width = images.shape
=======
) -> torch.Tensor:
    progress = (step + 0.5) / max(total_steps, 1)
    cross_offset_probability = 0.25 + 0.25 * progress
    batch, channels, height, width = images.shape
>>>>>>> REPLACE

<<<<<<< SEARCH
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.375
=======
    cross_offset_mask = (
        torch.rand(batch, device=images.device)
        < cross_offset_probability
    )
>>>>>>> REPLACE