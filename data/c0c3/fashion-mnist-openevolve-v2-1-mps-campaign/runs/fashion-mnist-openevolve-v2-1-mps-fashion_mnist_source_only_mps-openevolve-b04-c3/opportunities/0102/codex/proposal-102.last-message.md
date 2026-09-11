MECHANISM: Convex local inference-temperature refinement

HYPOTHESIS: Scaling ensemble logits by 1.035946 will preserve all 9,243 predictions while reducing validation cross-entropy below 0.20883300514221192.

INTENDED_EDIT: Change only the positive inference-time logit scale from 1.0360 to 1.035946.

EVIDENCE: Scales 1.035897, 1.03592, and 1.0360 yielded cross-entropies 0.20883300971984864, 0.20883300514221192, and 0.20883301086425782 with identical correct counts; quadratic interpolation of these closest measurements places the local minimum near 1.035946.

<<<<<<< SEARCH
            logits = 1.0360 * (
=======
            logits = 1.035946 * (
>>>>>>> REPLACE