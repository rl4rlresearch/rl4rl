MECHANISM: Fine-grained EMA boundary search

HYPOTHESIS: An EMA coefficient of 0.0325 will exceed 9,260 correct predictions by tracking slightly later states than the best 0.03 design without the accuracy loss observed at 0.035.

INTENDED_EDIT: Increase only the EMA interpolation coefficient from 0.03 to 0.0325.

EVIDENCE: EMA coefficients 0.01, 0.02, and 0.03 progressively improved validation_correct to 9,260, while 0.035 fell to 9,257; testing the midpoint isolates the apparent optimum boundary.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.03)
=======
                    average.lerp_(parameter.detach(), 0.0325)
>>>>>>> REPLACE