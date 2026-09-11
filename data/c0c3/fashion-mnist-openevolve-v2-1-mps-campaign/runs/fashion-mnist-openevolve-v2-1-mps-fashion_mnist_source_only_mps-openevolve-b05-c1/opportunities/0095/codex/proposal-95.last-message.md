MECHANISM: Evidence-guided classifier dropout interpolation

HYPOTHESIS: Setting classifier dropout to 0.1125 will exceed 9,345 correct predictions by moving modestly toward stronger regularization without incurring the degradation observed at 0.15.

INTENDED_EDIT: Increase classifier dropout probability from 0.10 to 0.1125; all other architecture, optimization, augmentation, and evaluation behavior remains unchanged.

EVIDENCE: Dropout 0.10 achieved 9,345 correct, while 0.05 fell to 9,301 and 0.15 retained 9,330; the asymmetric degradation indicates that the local optimum may lie slightly above 0.10.

<<<<<<< SEARCH
            nn.Dropout(0.1),
=======
            nn.Dropout(0.1125),
>>>>>>> REPLACE