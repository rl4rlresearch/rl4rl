MECHANISM: Warmup-free cosine decay

HYPOTHESIS: Removing the 5% learning-rate warmup will exceed 9,319 correct predictions by preserving aggressive early optimization throughout the short two-pass training horizon.

INTENDED_EDIT: Replace the warmup-plus-cosine schedule with cosine decay from the full 2.0e-3 learning rate to 3% of peak.

EVIDENCE: Making the first update warmup-consistent reduced correctness from 9,319 to 9,298; this suggests the full-rate initial update is beneficial and motivates testing full-rate early training rather than a 5% low-rate ramp.

<<<<<<< SEARCH
    progress = min(step / max(total_steps, 1), 1.0)
    if progress < 0.05:
        multiplier = 0.2 + 0.8 * progress / 0.05
    else:
        cosine_progress = (progress - 0.05) / 0.95
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
=======
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.03 + 0.97 * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
>>>>>>> REPLACE