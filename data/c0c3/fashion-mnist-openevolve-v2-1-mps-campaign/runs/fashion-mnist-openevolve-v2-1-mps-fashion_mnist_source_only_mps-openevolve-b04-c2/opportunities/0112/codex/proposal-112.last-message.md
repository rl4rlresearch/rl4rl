MECHANISM: Fine-grained dense BatchNorm momentum interpolation

HYPOTHESIS: BatchNorm1d momentum 0.2025 will exceed 9,295 correct predictions by interpolating between the two momentum settings that both achieved 9,295 correct.

INTENDED_EDIT: Change only the classifier BatchNorm1d momentum from 0.3 to 0.2025.

EVIDENCE: Momentum 0.2 and 0.205 each achieved 9,295 correct, outperforming nearby 0.1975 and 0.21; their untested midpoint is the most targeted remaining refinement.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.3),
=======
            nn.BatchNorm1d(128, momentum=0.2025),
>>>>>>> REPLACE