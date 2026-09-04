MECHANISM: Three-point local temperature interpolation

HYPOTHESIS: Scaling ensemble logits by 1.035897 will preserve all 9,243 predictions while reducing validation cross-entropy below 0.20883300514221192.

INTENDED_EDIT: Refine only the positive inference-time logit scale from 1.03592 to 1.035897.

EVIDENCE: The verified scales 1.03577, 1.03592, and 1.036 produced cross-entropies 0.20883301391601564, 0.20883300514221192, and 0.20883301086425782 with identical correct counts; local quadratic interpolation places the minimum near 1.035897.

<<<<<<< SEARCH
            logits = 1.03592 * (
=======
            logits = 1.035897 * (
>>>>>>> REPLACE