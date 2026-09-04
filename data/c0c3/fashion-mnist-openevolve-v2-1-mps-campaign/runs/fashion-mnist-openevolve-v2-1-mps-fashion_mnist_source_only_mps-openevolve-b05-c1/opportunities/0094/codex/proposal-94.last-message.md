MECHANISM: Stronger classifier co-adaptation control

HYPOTHESIS: Increasing classifier dropout from 0.10 to 0.15 will exceed 9,345 correct predictions by strengthening the regularization that the reduced-dropout result showed was beneficial.

INTENDED_EDIT: Raise the existing classifier dropout probability to 0.15 without changing parameters, optimizer steps, augmentation, or evaluation cost.

EVIDENCE: Lowering dropout to 0.05 reduced validation correct from 9,345 to 9,301 and worsened cross-entropy, directly motivating a controlled test in the opposite direction.

<<<<<<< SEARCH
            nn.Dropout(0.1),
=======
            nn.Dropout(0.15),
>>>>>>> REPLACE