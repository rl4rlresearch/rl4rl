MECHANISM: Correctness-boundary BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.19777162075042724609375 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.2028640983581543.

INTENDED_EDIT: Decrease only the classifier BatchNorm1d momentum from 0.19777164459228515625 to 0.19777162075042724609375.

EVIDENCE: Momentum 0.197771625518798828125 retained 9,295 correct, while 0.1977716159820556640625 produced 9,294; their exact midpoint is the most informative unresolved correctness-boundary test.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.19777164459228515625),
=======
            nn.BatchNorm1d(128, momentum=0.19777162075042724609375),
>>>>>>> REPLACE