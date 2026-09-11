MECHANISM: Dual-statistic mean–peak refinement gating

HYPOTHESIS: Restoring the qualified equal mean/max descriptor will increase validation correctness from 9,319 to approximately 9,322 while retaining the current parameter count and calibrated inference temperature.

INTENDED_EDIT: Condition the refinement gate on an equal blend of global average and maximum channel responses.

EVIDENCE: Reference Design 3 used this exact descriptor and achieved the best available result of 9,322 correct, outperforming the current average-only gate’s 9,319; reducing the maximum contribution to 40% fell to 9,297.

<<<<<<< SEARCH
        pooled = F.adaptive_avg_pool2d(refined, output_size=1)
=======
        pooled = 0.5 * (
            F.adaptive_avg_pool2d(refined, output_size=1)
            + F.adaptive_max_pool2d(refined, output_size=1)
        )
>>>>>>> REPLACE