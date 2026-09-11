MECHANISM: Fine-grained dense BatchNorm momentum refinement

HYPOTHESIS: BatchNorm1d momentum 0.19 will exceed 9,295 correct predictions by retaining the rapid adaptation seen at momentum 0.2 while slightly reducing running-statistic noise.

INTENDED_EDIT: Add classifier BatchNorm1d with momentum 0.19 and use the best-verified evaluation logit scale of 1.20514.

EVIDENCE: Momentum 0.2 achieved 9,295 correct, outperforming 9,288 at 0.1, while momentum 0.25 and 0.3 fell to 9,292; this brackets the optimum near 0.2 and motivates a small downward refinement.

<<<<<<< SEARCH
            nn.Linear(128 * 3 * 3, 128),
            nn.GELU(),
=======
            nn.Linear(128 * 3 * 3, 128),
            nn.BatchNorm1d(128, momentum=0.19),
            nn.GELU(),
>>>>>>> REPLACE

<<<<<<< SEARCH
            logits = 1.20485 * logits
=======
            logits = 1.20514 * logits
>>>>>>> REPLACE