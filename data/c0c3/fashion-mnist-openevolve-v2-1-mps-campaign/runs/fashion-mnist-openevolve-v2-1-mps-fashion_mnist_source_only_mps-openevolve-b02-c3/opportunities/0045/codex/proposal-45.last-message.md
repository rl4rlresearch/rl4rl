MECHANISM: Fixed robust mean–peak channel gating

HYPOTHESIS: Reducing the peak statistic’s fixed contribution from 0.50 to 0.40 will exceed 9,322 correct predictions by retaining localized evidence while limiting sensitivity to isolated activation maxima.

INTENDED_EDIT: Change the refinement-gate descriptor from an equal mean/max blend to a parameter-free 60% mean and 40% maximum blend.

EVIDENCE: Equal mean/max gating achieved the best result at 9,322 correct, while average-only gating reached 9,319 and learnable per-channel mixing fell to 9,300; the timed-out top-four experiment further motivates this computationally free approximation to a less dominant peak statistic.

<<<<<<< SEARCH
        pooled = 0.5 * (
            F.adaptive_avg_pool2d(refined, output_size=1)
            + F.adaptive_max_pool2d(refined, output_size=1)
        )
=======
        pooled = (
            0.6 * F.adaptive_avg_pool2d(refined, output_size=1)
            + 0.4 * F.adaptive_max_pool2d(refined, output_size=1)
        )
>>>>>>> REPLACE