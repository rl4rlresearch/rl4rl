MECHANISM: Three-point parabolic BatchNorm momentum interpolation

HYPOTHESIS: A classifier BatchNorm1d momentum of 0.197771655718485514323 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.2028640983581543.

INTENDED_EDIT: Change only the classifier BatchNorm1d momentum to the estimated local cross-entropy minimum.

EVIDENCE: Momentum 0.19777164459228515625 produced the best observed cross-entropy, while the bracketing 0.197771625518798828125 and 0.1977716827392578125 values retained 9,295 correct with higher cross-entropies; parabolic interpolation of those three results estimates a minimum slightly above the current best momentum.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.197771759033203125),
=======
            nn.BatchNorm1d(128, momentum=0.197771655718485514323),
>>>>>>> REPLACE