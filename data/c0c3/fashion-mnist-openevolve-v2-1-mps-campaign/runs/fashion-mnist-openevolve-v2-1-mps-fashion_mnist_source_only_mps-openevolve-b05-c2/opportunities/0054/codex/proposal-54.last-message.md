MECHANISM: Local quadratic order-preserving temperature refinement

HYPOTHESIS: An effective denominator of 76.512% will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22237594985961914.

INTENDED_EDIT: Replace the current 84/75.9 sharpening factor with 84/76.512 while retaining max-centering and class ordering.

EVIDENCE: Cross-entropy was 0.22238604049682617 at 75.9%, 0.2223825225830078 at 76%, and 0.22237594985961914 at 76.74%; a local quadratic through these three nearby measurements places the estimated minimum near 76.512%.

<<<<<<< SEARCH
        return normalized_logits * (280.0 / 253.0)
=======
        return normalized_logits * (10500.0 / 9564.0)
>>>>>>> REPLACE