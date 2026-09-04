MECHANISM: Endpoint-dwelling smoothstep supervision curriculum

HYPOTHESIS: A quintic smoothstep transition from 0.60 to 0.90 will exceed 9,239 correct predictions by preserving the successful endpoints and mean while spending more training time in the early per-view and late ensemble-specialized regimes that cosine scheduling improved.

INTENDED_EDIT: Replace the cosine ensemble-loss curriculum with a symmetric quintic smoothstep curriculum having identical endpoints, midpoint, and mean weight.

EVIDENCE: The 0.60–0.90 cosine curriculum improved correctness from 9,237 for the linear curriculum to 9,239, while changing the endpoint range regressed; increasing endpoint dwell without altering that supported range isolates the apparent benefit of temporal concentration.

<<<<<<< SEARCH
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
=======
    smooth_progress = progress**3 * (
        10.0 - 15.0 * progress + 6.0 * progress**2
    )
    ensemble_weight = 0.60 + 0.30 * smooth_progress
>>>>>>> REPLACE