MECHANISM: Layout-preserving dense-head expansion

HYPOTHESIS: Widening the classifier hidden layer from 40 to 44 units will exceed 9,290 correct predictions by increasing layout-sensitive capacity while remaining below the 250,000-parameter ceiling.

INTENDED_EDIT: Expand the successful flattening classifier to 44 hidden units, yielding 249,934 learned parameters while leaving training and regularization unchanged.

EVIDENCE: Reallocating dense-head parameters into an additional residual block reduced correctness, whereas the 40-unit layout-preserving head produced the strongest 9,290 result; this tests the opposite allocation direction using the remaining parameter budget.

<<<<<<< SEARCH
            nn.Linear(64 * 7 * 7, 40),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(40, 10),
=======
            nn.Linear(64 * 7 * 7, 44),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(44, 10),
>>>>>>> REPLACE